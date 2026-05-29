import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import repeat
from copy import deepcopy

from models.adapter import ResidualAdapter, TrajectoryPretextHead, next_position_loss


class Final_Model(nn.Module):
    """
    P-TTA version of PPT test-time trajectory model.

    This file is based on models/model_test_trajectory.py, not models/model.py.

    Why?
        - models/model.py is mainly used for training modes.
        - Its ALL-mode forward uses destination GT during training.
        - models/model_test_trajectory.py is the correct inference path because
          it predicts 20 future trajectories without future GT.

    Main additions:
        1. Frozen PPT backbone.
        2. Lightweight ResidualAdapter after traj_encoder(past).
        3. Auxiliary TrajectoryPretextHead for observed-past self-supervision.
        4. Helper methods for future test-time adaptation trainer.
    """

    def __init__(self, config, pretrained_model):
        super(Final_Model, self).__init__()

        self.name_model = 'PPT_Model_Test_PTTA'
        self.use_cuda = config.cuda
        self.dim_embedding_key = 128
        self.past_len = config.past_len
        self.future_len = config.future_len
        self.mode = config.mode

        assert self.mode == 'ALL', 'WRONG MODE! P-TTA test model expects mode == ALL.'

        # ---------------------------------------------------------
        # 1. Copy/load PPT pretrained modules
        # ---------------------------------------------------------
        self.Traj_encoder = deepcopy(pretrained_model.Traj_encoder)
        self.AR_Model = deepcopy(pretrained_model.AR_Model)
        self.predictor_Des = deepcopy(pretrained_model.predictor_Des)
        self.rand_token = deepcopy(pretrained_model.rand_token)
        self.des_encoder = deepcopy(pretrained_model.des_encoder)

        self.traj_trans_layer = pretrained_model.traj_trans_layer
        self.traj_encoder = pretrained_model.traj_encoder
        self.traj_rand_fut_token = pretrained_model.traj_rand_fut_token
        self.traj_fut_token_encoder = pretrained_model.traj_fut_token_encoder
        self.traj_decoder = pretrained_model.traj_decoder
        self.traj_decoder_9 = pretrained_model.traj_decoder_9
        self.traj_decoder_20 = pretrained_model.traj_decoder_20

        # ---------------------------------------------------------
        # 2. Freeze all PPT backbone parameters
        # ---------------------------------------------------------
        for p in self.parameters():
            p.requires_grad = False

        # ---------------------------------------------------------
        # 3. Add P-TTA trainable modules
        # ---------------------------------------------------------
        self.n_embd = getattr(config, "n_embd", 128)
        self.coord_dim = getattr(config, "vocab_size", 2)

        adapter_bottleneck = getattr(config, "ptta_adapter_bottleneck", 32)
        adapter_dropout = getattr(config, "ptta_adapter_dropout", 0.0)

        pretext_hidden_dim = getattr(config, "ptta_pretext_hidden_dim", 64)
        pretext_dropout = getattr(config, "ptta_pretext_dropout", 0.0)

        # This adapter is the lightweight trainable module for P-TTA.
        self.past_adapter = ResidualAdapter(
            dim=self.n_embd,
            bottleneck=adapter_bottleneck,
            dropout=adapter_dropout,
            use_actor_token=False
        )

        # Auxiliary head for observed-past self-supervision.
        # It can be used for MAE-style reconstruction or next-position prediction.
        self.pretext_head = TrajectoryPretextHead(
            dim=self.n_embd,
            hidden_dim=pretext_hidden_dim,
            out_dim=self.coord_dim,
            dropout=pretext_dropout
        )

        # Alias for readability in later trainer code.
        # We can use the same head for next-position prediction.
        self.pretext_next_head = self.pretext_head

        # Whether to also apply the adapter to the destination prediction branch.
        # Default is False for conservative behavior.
        # If later experiments show limited FDE improvement, this can be tested as an ablation.
        self.ptta_adapt_destination = getattr(config, "ptta_adapt_destination", False)

        # Make sure only P-TTA modules are trainable.
        self.freeze_ppt_backbone()

    # ---------------------------------------------------------
    # Parameter control utilities
    # ---------------------------------------------------------
    def freeze_ppt_backbone(self):
        """
        Freeze all pretrained PPT parameters and keep only P-TTA modules trainable.
        """
        for name, p in self.named_parameters():
            if name.startswith("past_adapter") or name.startswith("pretext_head"):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def get_tta_parameters(self, update_head=True):
        """
        Return parameters that will be updated during test-time adaptation.

        Args:
            update_head:
                If True, update adapter + pretext head.
                If False, update adapter only.

        Returns:
            List of trainable parameters.
        """
        params = list(self.past_adapter.parameters())

        if update_head:
            params += list(self.pretext_head.parameters())

        return [p for p in params if p.requires_grad]

    def print_trainable_parameters(self):
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print("[P-TTA Model]")
        print(f"Trainable/Total parameters: {trainable_num}/{total_num}")

        for name, p in self.named_parameters():
            if p.requires_grad:
                print(f"  trainable: {name} | shape={tuple(p.shape)}")

    # ---------------------------------------------------------
    # Feature extraction utilities for P-TTA
    # ---------------------------------------------------------
    def extract_past_feat(self, past):
        """
        Extract frozen PPT trajectory features before adapter.

        Args:
            past: [B, 8, 2]

        Returns:
            past_feat: [B, 8, D]
        """
        return self.traj_encoder(past)

    def extract_adapted_past_feat(self, past, actor_token=None):
        """
        Extract PPT trajectory features and pass them through the adapter.

        Args:
            past:
                Observed past trajectory, usually x_shift, shape [B, 8, 2].

            actor_token:
                Optional actor-specific token. Currently not used in the ETH/UCY pipeline.

        Returns:
            adapted_feat: [B, 8, D]
        """
        past_feat = self.extract_past_feat(past)
        adapted_feat = self.past_adapter(past_feat, actor_token=actor_token)
        return adapted_feat

    def reconstruct_past_from_feat(self, past, actor_token=None):
        """
        Predict coordinates from adapted past features.

        This is used by MAE-style observed-past reconstruction loss.

        Args:
            past: [B, 8, 2]

        Returns:
            pred_xy: [B, 8, 2]
        """
        feat = self.extract_adapted_past_feat(past, actor_token=actor_token)
        pred_xy = self.pretext_head(feat)
        return pred_xy

    def predict_next_from_past(self, past, actor_token=None):
        """
        PPT Task-I-style next-position prediction from observed past.

        Args:
            past: [B, 8, 2]

        Returns:
            pred_next: [B, 7, 2]
                       pred_next[:, t] predicts past[:, t+1].
        """
        feat = self.extract_adapted_past_feat(past, actor_token=actor_token)
        pred_next = self.pretext_next_head(feat[:, :-1, :])
        return pred_next
    
    def pretext_next_loss(
        self,
        past,
        valid_mask=None,
        loss_type="smooth_l1",
        actor_token=None,
        require_source_valid=False,
        return_stats=False,
    ):
        """
        Pretext-consistent next-position self-supervised loss for P-TTA.

        This loss reuses PPT Stage-I philosophy at test time:
            x1 -> x2
            x2 -> x3
            ...
            x7 -> x8

        Args:
            past:
                Observed past trajectory, shape [B, 8, 2].
                In robust/P-TTA evaluation this will usually be x_shift,
                i.e., the corrupted observed past actually seen by the model.

            valid_mask:
                Optional bool tensor, shape [B, 8].
                True means the point is actually available.
                False means the point was missing/dropped and artificially filled.

                Important:
                    If x3 is missing and filled, then x2 -> x3 should NOT
                    use x3 as a self-supervised target. Therefore we mask
                    target positions x2~x8 using valid_mask[:, 1:].

            loss_type:
                "smooth_l1", "l1", or "l2".

            actor_token:
                Optional actor-specific token. Currently unused in the ETH/UCY
                pipeline, but kept for future extension.

            require_source_valid:
                If False:
                    Use a transition as long as the target x_{t+1} is valid.
                    This matches the minimal PPT Task-I self-supervision rule.

                If True:
                    Use a transition only when both source x_t and target x_{t+1}
                    are valid. This is stricter and can be tested as an ablation.

            return_stats:
                If True, return (loss, stats_dict).

        Returns:
            loss:
                Scalar tensor.

            stats_dict, optional:
                Contains the number of valid transition targets.
        """
        pred_next = self.predict_next_from_past(
            past,
            actor_token=actor_token
        )  # [B, 7, 2]

        target_next = past[:, 1:, :]  # [B, 7, 2]

        if valid_mask is None:
            final_valid = torch.ones(
                target_next.shape[:2],
                dtype=torch.bool,
                device=target_next.device
            )
        else:
            if valid_mask.dtype != torch.bool:
                valid_mask = valid_mask.bool()

            # Target positions are x2~x8.
            final_valid = valid_mask[:, 1:]

            if require_source_valid:
                # Source positions are x1~x7.
                source_valid = valid_mask[:, :-1]
                final_valid = final_valid & source_valid

        if final_valid.sum() == 0:
            loss = pred_next.sum() * 0.0
        else:
            # Use the utility function from adapter.py.
            # It masks target positions using valid_mask[:, 1:].
            if not require_source_valid:
                loss = next_position_loss(
                    pred_next=pred_next,
                    target_next=target_next,
                    valid_mask=valid_mask,
                    loss_type=loss_type
                )
            else:
                # For the stricter source+target valid case, compute directly.
                if loss_type == "smooth_l1":
                    point_loss = F.smooth_l1_loss(
                        pred_next,
                        target_next,
                        reduction="none"
                    ).sum(dim=-1)
                elif loss_type == "l1":
                    point_loss = torch.abs(pred_next - target_next).sum(dim=-1)
                elif loss_type == "l2":
                    point_loss = torch.norm(pred_next - target_next, dim=-1)
                else:
                    raise ValueError(f"Unsupported loss_type: {loss_type}")

                loss = point_loss[final_valid].mean()

        if return_stats:
            stats = {
                "num_valid_next_targets": int(final_valid.sum().detach().cpu().item()),
                "num_total_next_targets": int(final_valid.numel()),
                "valid_next_ratio": float(
                    final_valid.float().mean().detach().cpu().item()
                ),
            }
            return loss, stats

        return loss

    # ---------------------------------------------------------
    # Destination prediction utility
    # ---------------------------------------------------------
    def predict_destinations(self, past):
        """
        Predict 20 destination candidates using PPT destination branch.

        Args:
            past: [B, 8, 2]

        Returns:
            destination_prediction: [B, 20, 2]
        """
        past_state = self.Traj_encoder(past)

        if self.ptta_adapt_destination:
            # Optional ablation:
            # Apply the same adapter to the destination branch feature.
            # Default is False because we first want a conservative P-TTA variant.
            past_state = self.past_adapter(past_state)

        des_token = repeat(self.rand_token, '() n d -> b n d', b=past.size(0))
        des_state = self.des_encoder(des_token)
        traj_state = torch.cat((past_state, des_state), dim=1)

        feat = self.AR_Model(traj_state)
        pred_des = self.predictor_Des(feat[:, -1])
        destination_prediction = pred_des.view(pred_des.size(0), 20, -1)

        return destination_prediction

    # ---------------------------------------------------------
    # Main test-time forward
    # ---------------------------------------------------------
    def forward(self, past, abs_past, seq_start_end, end_pose):
        """
        Generate 20 future trajectory candidates.

        Args:
            past:
                Relative observed past, shape [B, 8, 2].
                In robust/P-TTA evaluation this can be corrupted x_shift.

            abs_past:
                Absolute observed past, shape [B, 8, 2].
                In robust/P-TTA evaluation this should match past, i.e. abs_past_shift.

            seq_start_end:
                Social grouping information from PPT dataloader.

            end_pose:
                Last observed absolute position x8, shape [B, 1, 2].

        Returns:
            predictions:
                Future trajectory candidates, shape [B, 20, 12, 2].
        """
        destination_prediction = self.predict_destinations(past)

        predictions = []

        for i in range(20):
            fut_token = repeat(
                self.traj_rand_fut_token,
                '() n d -> b n d',
                b=past.size(0)
            )

            # ---------------------------------------------------------
            # P-TTA insertion point:
            # Original PPT:
            #     past_feat = self.traj_encoder(past)
            #
            # P-TTA:
            #     past_feat = self.traj_encoder(past)
            #     past_feat = self.past_adapter(past_feat)
            # ---------------------------------------------------------
            past_feat = self.extract_adapted_past_feat(past)

            fut_feat = self.traj_fut_token_encoder(fut_token)
            des_feat = self.traj_encoder(destination_prediction[:, i])

            traj_feat = torch.cat(
                (past_feat, fut_feat, des_feat.unsqueeze(1)),
                dim=1
            )

            prediction_feat = self.traj_trans_layer(traj_feat, mask_type='all')

            pre_prediction = self.traj_decoder_9(
                prediction_feat[:, self.past_len - 1:self.past_len]
            )

            mid_prediction = self.traj_decoder(
                prediction_feat[:, self.past_len:-2]
            )

            des_prediction = self.traj_decoder_20(
                prediction_feat[:, -2:-1]
            ) + destination_prediction[:, i].unsqueeze(1)

            total_prediction = torch.cat(
                (pre_prediction, mid_prediction, des_prediction),
                dim=1
            )

            predictions.append(total_prediction.unsqueeze(1))

        predictions = torch.cat(predictions, dim=1)

        return predictions