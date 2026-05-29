import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualAdapter(nn.Module):
    """
    Lightweight residual adapter for PPT-based P-TTA.

    This module is inserted after PPT's trajectory encoder:

        past_feat = self.traj_encoder(past)
        past_feat = self.past_adapter(past_feat)

    Input:
        x: [B, T, D]

    Output:
        y: [B, T, D]

    Important:
        The final linear layer is zero-initialized.
        Therefore, at initialization:

            adapter(x) = x + net(x) ≈ x

        This means adding the adapter does not immediately damage
        the pretrained PPT behavior.
    """

    def __init__(self, dim=128, bottleneck=32, dropout=0.0, use_actor_token=False):
        super().__init__()

        self.dim = dim
        self.bottleneck = bottleneck
        self.use_actor_token = use_actor_token

        if use_actor_token:
            self.actor_proj = nn.Linear(dim, dim)
        else:
            self.actor_proj = None

        layers = [
            nn.Linear(dim, bottleneck),
            nn.ReLU(inplace=True),
        ]

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(bottleneck, dim))

        self.net = nn.Sequential(*layers)

        # Zero initialization makes the adapter start as an identity mapping.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, actor_token=None):
        """
        Args:
            x:
                Feature tensor, shape [B, T, D].

            actor_token:
                Optional actor-specific token, shape [B, D] or [B, 1, D].
                This is inspired by T4P's actor-specific token memory.
                In the current PPT/ETH-UCY pipeline, actor IDs are not yet
                passed through the dataloader, so this argument can remain None.

        Returns:
            Adapted feature tensor, shape [B, T, D].
        """
        h = x

        if self.use_actor_token and actor_token is not None:
            if actor_token.dim() == 2:
                actor_token = actor_token.unsqueeze(1)  # [B, 1, D]

            h = h + self.actor_proj(actor_token)

        return x + self.net(h)


class TrajectoryPretextHead(nn.Module):
    """
    Coordinate reconstruction head for test-time self-supervision.

    This head predicts 2D coordinates from adapted PPT features.

    Usage examples:
        1. MAE-style masked reconstruction:
            pred_xy = head(adapted_feat)        # [B, T, 2]
            loss = masked_reconstruction_loss(pred_xy, target_xy, mae_mask)

        2. PPT Task-I-style next-position prediction:
            pred_next = head(adapted_feat[:, :-1])
            target_next = past[:, 1:]

    This is not part of the frozen PPT predictor.
    It is an auxiliary self-supervised head used to update the adapter
    during offline preparation or test-time adaptation.
    """

    def __init__(self, dim=128, hidden_dim=64, out_dim=2, dropout=0.0):
        super().__init__()

        layers = [
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
        ]

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, feat):
        """
        Args:
            feat: [B, T, D]

        Returns:
            pred_xy: [B, T, 2]
        """
        return self.net(feat)


def make_mae_mask(valid_mask, mask_ratio=0.3, keep_last=True, min_mask=1):
    """
    Create a random MAE-style mask over available observed past points.

    This is the key utility for T4P-inspired test-time self-supervision.

    We do NOT use future ground truth.
    We only mask part of the observed past and ask the model to reconstruct it.

    Args:
        valid_mask:
            Bool tensor, shape [B, T].
            True means the point is available.
            False means the point is already missing/dropped due to corruption.

        mask_ratio:
            Ratio of available observed points to mask for self-supervision.

        keep_last:
            If True, never mask the last observed point x8.
            This matches our PPT setting because x8 is the normalization anchor.

        min_mask:
            Minimum number of points to mask per sample if candidates exist.

    Returns:
        mae_mask:
            Bool tensor, shape [B, T].
            True means this point is masked and used as reconstruction target.
    """
    if valid_mask.dtype != torch.bool:
        valid_mask = valid_mask.bool()

    B, T = valid_mask.shape
    device = valid_mask.device

    mae_mask = torch.zeros(B, T, dtype=torch.bool, device=device)

    for b in range(B):
        candidate = torch.where(valid_mask[b])[0]

        if keep_last:
            candidate = candidate[candidate != (T - 1)]

        num_candidate = candidate.numel()

        if num_candidate == 0:
            continue

        num_mask = int(round(float(num_candidate) * float(mask_ratio)))
        num_mask = max(min_mask, num_mask)
        num_mask = min(num_mask, num_candidate)

        perm = torch.randperm(num_candidate, device=device)
        selected = candidate[perm[:num_mask]]

        mae_mask[b, selected] = True

    return mae_mask


def apply_mae_mask_to_past(past, mae_mask, fill="zero", noise_std=0.01):
    """
    Apply an MAE-style mask to observed past coordinates.

    Args:
        past:
            Observed past trajectory, shape [B, T, 2].
            Usually this is x_shift, i.e., corrupted past after robustness shift.

        mae_mask:
            Bool tensor, shape [B, T].
            True means this point should be hidden from the input and reconstructed.

        fill:
            How to hide masked points.
            Supported:
                - "zero": replace masked points with 0
                - "last": replace masked points with nearest previous available value
                - "noise": add Gaussian noise to masked points

        noise_std:
            Used only when fill == "noise".

    Returns:
        masked_past:
            Past trajectory after applying MAE mask, shape [B, T, 2].
    """
    if mae_mask.dtype != torch.bool:
        mae_mask = mae_mask.bool()

    masked_past = past.clone()
    B, T, D = past.shape
    device = past.device

    if fill == "zero":
        masked_past[mae_mask] = 0.0
        return masked_past

    if fill == "noise":
        noise = torch.randn_like(masked_past) * float(noise_std)
        masked_past[mae_mask] = masked_past[mae_mask] + noise[mae_mask]
        return masked_past

    if fill == "last":
        original = past.clone()

        for b in range(B):
            masked_indices = torch.where(mae_mask[b])[0]

            for idx_tensor in masked_indices:
                idx = int(idx_tensor.item())
                fill_value = None

                # Find nearest unmasked point on the left.
                for left in range(idx - 1, -1, -1):
                    if not mae_mask[b, left]:
                        fill_value = original[b, left]
                        break

                # If no left point exists, find nearest unmasked point on the right.
                if fill_value is None:
                    for right in range(idx + 1, T):
                        if not mae_mask[b, right]:
                            fill_value = original[b, right]
                            break

                if fill_value is None:
                    fill_value = torch.zeros(D, device=device, dtype=past.dtype)

                masked_past[b, idx] = fill_value

        return masked_past

    raise ValueError(f"Unsupported fill method: {fill}")


def masked_reconstruction_loss(pred, target, mae_mask, valid_mask=None, loss_type="smooth_l1"):
    """
    Compute reconstruction loss only on MAE-masked target points.

    This is the core self-supervised loss for P-TTA.

    Args:
        pred:
            Predicted coordinates, shape [B, T, 2].

        target:
            Target coordinates, shape [B, T, 2].
            This should be observed past coordinates, not future ground truth.

        mae_mask:
            Bool tensor, shape [B, T].
            True means this point is reconstructed and contributes to loss.

        valid_mask:
            Optional bool tensor, shape [B, T].
            True means this point is actually available in the observation.
            If provided, loss is computed only where mae_mask & valid_mask is True.

        loss_type:
            "smooth_l1", "l1", or "l2".

    Returns:
        loss:
            Scalar tensor.
    """
    if mae_mask.dtype != torch.bool:
        mae_mask = mae_mask.bool()

    final_mask = mae_mask

    if valid_mask is not None:
        if valid_mask.dtype != torch.bool:
            valid_mask = valid_mask.bool()
        final_mask = final_mask & valid_mask

    if final_mask.sum() == 0:
        # Return a zero scalar that still lives on the correct device.
        return pred.sum() * 0.0

    if loss_type == "smooth_l1":
        point_loss = F.smooth_l1_loss(pred, target, reduction="none").sum(dim=-1)
    elif loss_type == "l1":
        point_loss = torch.abs(pred - target).sum(dim=-1)
    elif loss_type == "l2":
        point_loss = torch.norm(pred - target, dim=-1)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    loss = point_loss[final_mask].mean()
    return loss


def next_position_loss(pred_next, target_next, valid_mask=None, loss_type="smooth_l1"):
    """
    PPT Task-I-style next-position self-supervision.

    Args:
        pred_next:
            Predicted next positions, shape [B, T-1, 2].

        target_next:
            Target next positions from observed past, shape [B, T-1, 2].

        valid_mask:
            Optional bool tensor, shape [B, T].
            If provided, target positions x2~x8 are used only when valid.

        loss_type:
            "smooth_l1", "l1", or "l2".

    Returns:
        loss:
            Scalar tensor.
    """
    if valid_mask is not None:
        if valid_mask.dtype != torch.bool:
            valid_mask = valid_mask.bool()
        target_valid = valid_mask[:, 1:]
    else:
        target_valid = torch.ones(
            target_next.shape[:2],
            dtype=torch.bool,
            device=target_next.device
        )

    if target_valid.sum() == 0:
        return pred_next.sum() * 0.0

    if loss_type == "smooth_l1":
        point_loss = F.smooth_l1_loss(pred_next, target_next, reduction="none").sum(dim=-1)
    elif loss_type == "l1":
        point_loss = torch.abs(pred_next - target_next).sum(dim=-1)
    elif loss_type == "l2":
        point_loss = torch.norm(pred_next - target_next, dim=-1)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    return point_loss[target_valid].mean()


class ActorTokenMemory:
    """
    Simplified actor-specific token memory inspired by T4P.

    Important:
        This is only a scaffold for the current PPT project.

        The current PPT ETH/UCY evaluation loop does not explicitly pass
        persistent actor IDs to the trainer. Therefore, this memory should
        not be activated until the dataloader or evaluation loop provides
        stable actor IDs.

    Typical future usage:
        memory.update(actor_ids, actor_tokens)
        token = memory.get(actor_ids, default_token)

    This class stores detached tokens, so it does not backpropagate through memory.
    """

    def __init__(self, dim=128, momentum=0.9, device=None):
        self.dim = dim
        self.momentum = momentum
        self.device = device
        self.storage = {}

    def reset(self):
        self.storage = {}

    def get(self, actor_ids, default_token):
        """
        Args:
            actor_ids:
                List/Tensor of actor IDs, length B.

            default_token:
                Tensor, shape [B, D].

        Returns:
            memory_tokens:
                Tensor, shape [B, D].
        """
        if torch.is_tensor(actor_ids):
            actor_ids = actor_ids.detach().cpu().tolist()

        tokens = []

        for i, actor_id in enumerate(actor_ids):
            key = str(actor_id)

            if key in self.storage:
                tokens.append(self.storage[key].to(default_token.device))
            else:
                tokens.append(default_token[i].detach())

        return torch.stack(tokens, dim=0)

    def update(self, actor_ids, new_tokens):
        """
        Update memory with new actor tokens.

        Args:
            actor_ids:
                List/Tensor of actor IDs, length B.

            new_tokens:
                Tensor, shape [B, D].
        """
        if torch.is_tensor(actor_ids):
            actor_ids = actor_ids.detach().cpu().tolist()

        new_tokens = new_tokens.detach()

        for i, actor_id in enumerate(actor_ids):
            key = str(actor_id)
            token = new_tokens[i].detach().cpu()

            if key not in self.storage:
                self.storage[key] = token
            else:
                self.storage[key] = (
                    self.momentum * self.storage[key]
                    + (1.0 - self.momentum) * token
                )