import os
import datetime
import csv

import numpy as np
import torch
import torch.nn as nn
from models.model_test_trajectory_ptta import Final_Model
from torch.utils.data import DataLoader

from dataset_loader import *
from vis_ethucy import *

torch.set_num_threads(5)


class Trainer:
    def __init__(self, config):

        self.config = config
        self.device = torch.device('cuda') if config.cuda else torch.device('cpu')

        # test folder creating
        self.name_test = str(datetime.datetime.now())[:10]
        self.folder_test = 'testing/' + self.name_test + '_' + config.info
        if not os.path.exists(self.folder_test):
            os.makedirs(self.folder_test)
        self.folder_test = self.folder_test + '/'

        # Initialize dataset
        if config.dataset_name == 'sdd':
            data_folder = 'data'
            test_dataset = SocialDataset(
                data_folder,
                set_name="test",
                b_size=4096,
                t_tresh=0,
                d_tresh=100,
                scene='sdd'
            )
        elif config.dataset_name == 'eth':
            data_folder = 'data/ETH_UCY'
            test_dataset = SocialDataset(
                data_folder,
                set_name="test",
                b_size=4096,
                t_tresh=0,
                d_tresh=50,
                scene=config.data_scene
            )
        else:
            raise ValueError(f"Unsupported dataset_name: {config.dataset_name}")

        if config.vis and config.dataset_name == 'eth':
            self.homo_mat = {}
            for scene in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
                self.homo_mat[scene] = np.loadtxt(f'./data/ETH_image/{scene}_H.txt')

        # Initialize dataloader
        self.test_dataset = DataLoader(
            test_dataset,
            batch_size=1,
            collate_fn=socialtraj_collate
        )
        print('Loaded data!')

        if torch.cuda.is_available() and config.cuda:
            torch.cuda.set_device(config.gpu)

        self.settings = {
            "train_batch_size": config.train_b_size,
            "test_batch_size": config.test_b_size,
            "use_cuda": config.cuda,
            "dim_feature_tracklet": config.past_len * 2,
            "dim_feature_future": config.future_len * 2,
            "dim_embedding_key": config.dim_embedding_key,
            "past_len": config.past_len,
            "future_len": 12,
        }

        config.mode = 'ALL'

        # ---------------------------------------------------------
        # Load pretrained PPT model and build P-TTA model
        # ---------------------------------------------------------
        self.model_Pretrain = torch.load(
            config.model_Pretrain,
            map_location=torch.device('cpu')
        )

        if config.cuda:
            self.model_Pretrain = self.model_Pretrain.cuda()

        self.model = Final_Model(config, self.model_Pretrain)

        if config.cuda:
            self.model = self.model.cuda()

        # ---------------------------------------------------------
        # Load offline-initialized P-TTA adapter/head if provided
        # ---------------------------------------------------------
        if getattr(config, "ptta_init_checkpoint", ""):
            print(f"[P-TTA] Loading init checkpoint: {config.ptta_init_checkpoint}")
            self.model.load_ptta_state(
                config.ptta_init_checkpoint,
                map_location=self.device,
                strict=True
            )
        else:
            print("[P-TTA] No init checkpoint provided. Using randomly initialized adapter/head.")

        # Make sure PPT backbone is frozen after loading P-TTA state.
        self.model.freeze_ppt_backbone()

        # Store initial P-TTA state.
        # This state is restored before adapting each test batch.
        self.ptta_initial_state = self._clone_ptta_state()
        print("[P-TTA] Stored initial adapter/head state for per-batch reset.")

        self.start_epoch = 0

    def print_model_param(self, model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("\033[1;31;40mTrainable/Total: {}/{}\033[0m".format(trainable_num, total_num))
        return 0

    @staticmethod
    def _to_float(value):
        """
        Safely convert Python number or torch scalar tensor to float.
        This avoids issues when metrics are CUDA tensors.
        """
        if torch.is_tensor(value):
            return float(value.detach().cpu().item())
        return float(value)

    def _clone_ptta_state(self):
        """
        Clone only P-TTA module states.

        We reset to this state before every test batch so that each sample/batch
        is adapted independently.
        """
        return {
            "past_adapter": {
                k: v.detach().clone()
                for k, v in self.model.past_adapter.state_dict().items()
            },
            "pretext_head": {
                k: v.detach().clone()
                for k, v in self.model.pretext_head.state_dict().items()
            },
        }

    def _reset_ptta_state(self):
        """
        Reset adapter/head to the stored initial state.
        """
        self.model.past_adapter.load_state_dict(self.ptta_initial_state["past_adapter"])
        self.model.pretext_head.load_state_dict(self.ptta_initial_state["pretext_head"])

    def _set_tta_trainable_params(self):
        """
        Freeze PPT backbone and enable gradients for adapter only by default.

        If --tta_update_head is set, also update pretext_head.
        """
        for p in self.model.parameters():
            p.requires_grad = False

        for p in self.model.past_adapter.parameters():
            p.requires_grad = True

        if getattr(self.config, "tta_update_head", False):
            for p in self.model.pretext_head.parameters():
                p.requires_grad = True

    def _get_tta_params(self):
        """
        Return parameters updated at test time.
        """
        params = list(self.model.past_adapter.parameters())

        if getattr(self.config, "tta_update_head", False):
            params += list(self.model.pretext_head.parameters())

        return [p for p in params if p.requires_grad]

    def _set_tta_mode(self):
        """
        Keep frozen PPT deterministic and train only P-TTA modules.

        Do not call self.model.train() globally because frozen PPT may contain
        dropout. We want only adapter/head behavior to be trainable.
        """
        self.model.eval()
        self.model.past_adapter.train()

        if getattr(self.config, "tta_update_head", False):
            self.model.pretext_head.train()
        else:
            self.model.pretext_head.eval()

    def _run_test_time_update(self, x_shift, valid_mask):
        """
        Run P-TTA self-supervised update on a single test batch.

        Args:
            x_shift:
                Corrupted observed past, shape [B, 8, 2].

            valid_mask:
                Bool mask, shape [B, 8].
                True means observed/available.
                False means missing/drop filled.

        Returns:
            stats:
                Dictionary containing TTA loss information.
        """
        tta_steps = int(getattr(self.config, "tta_steps", 1))
        tta_lr = float(getattr(self.config, "tta_lr", 1e-4))
        grad_clip = float(getattr(self.config, "tta_grad_clip", 1.0))

        if tta_steps <= 0:
            return {
                "tta_loss_first": 0.0,
                "tta_loss_last": 0.0,
                "tta_steps": 0,
                "tta_lr": tta_lr,
                "valid_next_ratio": 0.0,
            }

        self._set_tta_trainable_params()
        self._set_tta_mode()

        optimizer_tta = torch.optim.Adam(
            self._get_tta_params(),
            lr=tta_lr
        )

        loss_values = []
        valid_ratios = []

        for _ in range(tta_steps):
            optimizer_tta.zero_grad()

            loss_tta, stats = self.model.pretext_next_loss(
                past=x_shift,
                valid_mask=valid_mask,
                loss_type=getattr(self.config, "tta_loss_type", "smooth_l1"),
                require_source_valid=getattr(self.config, "tta_require_source_valid", False),
                return_stats=True
            )

            loss_tta.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self._get_tta_params(),
                    grad_clip,
                    norm_type=2
                )

            optimizer_tta.step()

            loss_values.append(float(loss_tta.detach().cpu().item()))
            valid_ratios.append(float(stats["valid_next_ratio"]))

        self.model.eval()

        return {
            "tta_loss_first": loss_values[0],
            "tta_loss_last": loss_values[-1],
            "tta_steps": tta_steps,
            "tta_lr": tta_lr,
            "valid_next_ratio": sum(valid_ratios) / max(len(valid_ratios), 1),
        }

    def apply_observation_shift(self, x):
        """
        Apply controlled test-time observation corruption to the observed past trajectory.

        Args:
            x: normalized observed past trajectory, shape [B, T=8, 2].
               In PPT, x8 is the normalization anchor and becomes approximately [0, 0].

        Returns:
            x_shift: corrupted observed past, shape [B, 8, 2]
            valid_mask: bool mask, shape [B, 8].
                        True means this observed point is available.
                        False means this point is treated as missing/dropped.
        """
        shift_type = getattr(self.config, "shift_type", "clean")

        x_shift = x.clone()
        B, T, D = x_shift.shape
        device = x_shift.device

        valid_mask = torch.ones(B, T, dtype=torch.bool, device=device)

        # Important:
        # Keep x8 unchanged because PPT normalizes trajectories by the last observed point.
        # Candidate indices are x1~x7 only.
        candidate_indices = torch.arange(0, T - 1, device=device)

        if shift_type == "clean":
            return x_shift, valid_mask

        if shift_type == "missing":
            missing_rate = float(getattr(self.config, "missing_rate", 0.0))
            missing_pattern = getattr(self.config, "missing_pattern", "random")
            missing_fill = getattr(self.config, "missing_fill", "last")

            if missing_rate <= 0:
                return x_shift, valid_mask

            num_missing = int(round((T - 1) * missing_rate))
            num_missing = max(1, min(num_missing, T - 1))

            # First decide missing indices for each sample.
            missing_indices_per_sample = []

            for b in range(B):
                if missing_pattern == "random":
                    perm = candidate_indices[torch.randperm(len(candidate_indices), device=device)]
                    miss_idx = perm[:num_missing]
                elif missing_pattern == "consecutive":
                    max_start = (T - 1) - num_missing
                    start = torch.randint(0, max_start + 1, (1,), device=device).item()
                    miss_idx = torch.arange(start, start + num_missing, device=device)
                else:
                    raise ValueError(f"Unsupported missing_pattern: {missing_pattern}")

                valid_mask[b, miss_idx] = False
                missing_indices_per_sample.append(miss_idx)

            # Then fill missing points without using missing target coordinates.
            # We only use available observed points according to valid_mask.
            x_original = x.clone()

            for b in range(B):
                miss_idx = missing_indices_per_sample[b]

                for idx_tensor in miss_idx:
                    idx = int(idx_tensor.item())

                    if missing_fill == "zero":
                        x_shift[b, idx] = 0.0

                    elif missing_fill == "last":
                        # Use nearest available point on the left.
                        # If no left point exists, use nearest available point on the right.
                        fill_value = None

                        for left in range(idx - 1, -1, -1):
                            if valid_mask[b, left]:
                                fill_value = x_original[b, left]
                                break

                        if fill_value is None:
                            for right in range(idx + 1, T):
                                if valid_mask[b, right]:
                                    fill_value = x_original[b, right]
                                    break

                        if fill_value is None:
                            fill_value = torch.zeros(D, device=device, dtype=x.dtype)

                        x_shift[b, idx] = fill_value

                    elif missing_fill == "linear":
                        # Use nearest available left and right points.
                        left_value = None
                        right_value = None

                        for left in range(idx - 1, -1, -1):
                            if valid_mask[b, left]:
                                left_value = x_original[b, left]
                                break

                        for right in range(idx + 1, T):
                            if valid_mask[b, right]:
                                right_value = x_original[b, right]
                                break

                        if left_value is not None and right_value is not None:
                            x_shift[b, idx] = 0.5 * (left_value + right_value)
                        elif left_value is not None:
                            x_shift[b, idx] = left_value
                        elif right_value is not None:
                            x_shift[b, idx] = right_value
                        else:
                            x_shift[b, idx] = 0.0

                    else:
                        raise ValueError(f"Unsupported missing_fill: {missing_fill}")

            return x_shift, valid_mask

        if shift_type == "noise":
            noise_alpha = float(getattr(self.config, "noise_alpha", 0.0))

            if noise_alpha <= 0:
                return x_shift, valid_mask

            # Average displacement scale per sample.
            # Noise std = noise_alpha * average observed step displacement.
            step_disp = torch.norm(x[:, 1:] - x[:, :-1], dim=-1)  # [B, 7]
            step_scale = step_disp.mean(dim=1, keepdim=True).unsqueeze(-1)  # [B, 1, 1]

            noise = torch.randn_like(x_shift) * noise_alpha * step_scale

            # Keep x8 unchanged.
            noise[:, -1, :] = 0.0

            x_shift = x_shift + noise
            return x_shift, valid_mask

        if shift_type == "frame_drop":
            drop_num = int(getattr(self.config, "drop_num", 0))

            if drop_num <= 0:
                return x_shift, valid_mask

            drop_num = max(1, min(drop_num, T - 1))
            x_original = x.clone()

            for b in range(B):
                perm = candidate_indices[torch.randperm(len(candidate_indices), device=device)]
                drop_idx = perm[:drop_num]
                valid_mask[b, drop_idx] = False

                for idx_tensor in drop_idx:
                    idx = int(idx_tensor.item())

                    # Frame drop is simulated by holding the nearest previous available point.
                    fill_value = None

                    for left in range(idx - 1, -1, -1):
                        if valid_mask[b, left]:
                            fill_value = x_original[b, left]
                            break

                    if fill_value is None:
                        for right in range(idx + 1, T):
                            if valid_mask[b, right]:
                                fill_value = x_original[b, right]
                                break

                    if fill_value is None:
                        fill_value = torch.zeros(D, device=device, dtype=x.dtype)

                    x_shift[b, idx] = fill_value

            return x_shift, valid_mask

        raise ValueError(f"Unsupported shift_type: {shift_type}")

    def fit(self):

        dict_metrics_test = self.evaluate(self.test_dataset)

        ade = self._to_float(dict_metrics_test["ade_48s"])
        fde = self._to_float(dict_metrics_test["fde_48s"])

        print('[P-TTA Eval]')
        print('ADE_48s: {:.6f} | FDE_48s: {:.6f}'.format(ade, fde))
        print('shift_type: {} | missing_rate: {} | missing_pattern: {} | missing_fill: {} | noise_alpha: {} | drop_num: {}'.format(
            getattr(self.config, "shift_type", "clean"),
            getattr(self.config, "missing_rate", 0.0),
            getattr(self.config, "missing_pattern", "random"),
            getattr(self.config, "missing_fill", "last"),
            getattr(self.config, "noise_alpha", 0.0),
            getattr(self.config, "drop_num", 0),
        ))
        print('tta_steps: {} | tta_lr: {} | tta_loss_type: {} | tta_update_head: {} | tta_require_source_valid: {}'.format(
            getattr(self.config, "tta_steps", 1),
            getattr(self.config, "tta_lr", 1e-4),
            getattr(self.config, "tta_loss_type", "smooth_l1"),
            getattr(self.config, "tta_update_head", False),
            getattr(self.config, "tta_require_source_valid", False),
        ))
        print('-' * 100)

        # ---------------------------------------------------------
        # Save P-TTA evaluation result to CSV
        # ---------------------------------------------------------
        os.makedirs("results/ptta", exist_ok=True)

        csv_path = os.path.join("results", "ptta", f"{self.config.data_scene}.csv")
        write_header = not os.path.exists(csv_path)

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            if write_header:
                writer.writerow([
                    "timestamp",
                    "dataset_name",
                    "data_scene",
                    "model_path",
                    "past_len",
                    "future_len",
                    "ade_48s",
                    "fde_48s",
                    "shift_type",
                    "missing_rate",
                    "missing_pattern",
                    "missing_fill",
                    "noise_alpha",
                    "drop_num",
                    "info",
                    "robust_tag",
                    "tta_steps",
                    "tta_lr",
                    "tta_loss_type",
                    "tta_update_head",
                    "tta_require_source_valid",
                    "ptta_init_checkpoint",
                    "ptta_tag"
                ])

            writer.writerow([
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                self.config.dataset_name,
                self.config.data_scene,
                self.config.model_Pretrain,
                self.config.past_len,
                self.config.future_len,
                ade,
                fde,
                getattr(self.config, "shift_type", "clean"),
                getattr(self.config, "missing_rate", 0.0),
                getattr(self.config, "missing_pattern", "random"),
                getattr(self.config, "missing_fill", "last"),
                getattr(self.config, "noise_alpha", 0.0),
                getattr(self.config, "drop_num", 0),
                self.config.info,
                getattr(self.config, "robust_tag", "robust"),
                getattr(self.config, "tta_steps", 1),
                getattr(self.config, "tta_lr", 1e-4),
                getattr(self.config, "tta_loss_type", "smooth_l1"),
                getattr(self.config, "tta_update_head", False),
                getattr(self.config, "tta_require_source_valid", False),
                getattr(self.config, "ptta_init_checkpoint", ""),
                getattr(self.config, "ptta_tag", "ptta")
            ])

        print(f"[Saved P-TTA CSV] {csv_path}")

    def evaluate(self, dataset):

        ade_48s = fde_48s = 0
        samples = 0
        dict_metrics = {}

        self.model.eval()

        # ---------------------------------------------------------
        # Containers for visualization used by original PPT code
        # ---------------------------------------------------------
        past_gt = []
        fut_gt = []
        fut_pred_20 = []
        fut_pred_best = []

        # ---------------------------------------------------------
        # Containers for per-sample prediction saving
        # ---------------------------------------------------------
        save_enabled = getattr(self.config, "save_preds", False)

        save_past_rel = []
        save_future_rel = []
        save_pred20_rel = []
        save_best_ade_rel = []
        save_best_fde_rel = []

        save_past_abs = []
        save_future_abs = []
        save_pred20_abs = []
        save_best_ade_abs = []
        save_best_fde_abs = []

        # Clean original past and valid mask are important for robust/P-TTA analysis.
        save_past_rel_clean = []
        save_past_abs_clean = []
        save_valid_mask = []

        # P-TTA diagnostic stats.
        save_tta_loss_first = []
        save_tta_loss_last = []
        save_tta_valid_next_ratio = []

        save_ade_index = []
        save_fde_index = []
        save_sample_id = []
        save_scene = []

        sample_offset = 0

        # Important:
        # Do NOT wrap the whole evaluation loop with torch.no_grad().
        # P-TTA needs gradients during _run_test_time_update().
        for _, (trajectory, mask, initial_pos, seq_start_end) in enumerate(dataset):

            trajectory = torch.FloatTensor(trajectory).to(self.device)
            mask = torch.FloatTensor(mask).to(self.device)
            initial_pos = torch.FloatTensor(initial_pos).to(self.device)

            # ---------------------------------------------------------
            # PPT official evaluation uses coordinates normalized by
            # subtracting the last observed point x8.
            # ---------------------------------------------------------
            traj_norm = trajectory - trajectory[:, self.config.past_len - 1:self.config.past_len, :]

            x = traj_norm[:, :self.config.past_len, :]
            destination = traj_norm[:, -1:, :]
            y = traj_norm[:, self.config.past_len:, :]
            gt = traj_norm[:, 1:self.config.past_len + 1, :]

            abs_past = trajectory[:, :self.config.past_len, :]
            initial_pose = trajectory[:, self.config.past_len - 1:self.config.past_len, :]

            # ---------------------------------------------------------
            # Apply test-time observation corruption to observed past only.
            # x8 is kept unchanged inside apply_observation_shift().
            # ---------------------------------------------------------
            x_shift, valid_mask = self.apply_observation_shift(x)

            # The P-TTA/PPT model also receives abs_past.
            # Therefore, abs_past must be corrupted consistently with x_shift.
            # Since x_shift is relative to initial_pose, convert it back to absolute coordinates.
            abs_past_shift = x_shift + initial_pose

            # ---------------------------------------------------------
            # P-TTA: reset adapter/head to offline init state for this batch,
            # then update adapter using observed-past self-supervision.
            # ---------------------------------------------------------
            self._reset_ptta_state()

            tta_stats = self._run_test_time_update(
                x_shift=x_shift,
                valid_mask=valid_mask
            )

            # ---------------------------------------------------------
            # Predict future with updated adapter.
            # Future GT is not used during TTA update.
            # ---------------------------------------------------------
            self.model.eval()
            with torch.no_grad():
                output = self.model(x_shift, abs_past_shift, seq_start_end, initial_pose)
                output = output.data

                # ---------------------------------------------------------
                # Compute ADE/FDE using best-of-20 protocol.
                # The future GT remains clean and unchanged.
                # ---------------------------------------------------------
                future_rep = traj_norm[:, self.config.past_len:, :].unsqueeze(1).repeat(1, 20, 1, 1)
                distances = torch.norm(output - future_rep, dim=3)  # [B, 20, 12]

                # FDE: choose trajectory by final frame distance
                fde_mean_distances = torch.mean(distances[:, :, -1:], dim=2)
                fde_index_min = torch.argmin(fde_mean_distances, dim=1)
                fde_min_distances = distances[
                    torch.arange(0, len(fde_index_min), device=output.device),
                    fde_index_min
                ]
                fde_48s += torch.sum(fde_min_distances[:, -1])

                # ADE: choose trajectory by average distance over all future frames
                ade_mean_distances = torch.mean(distances[:, :, :], dim=2)
                ade_index_min = torch.argmin(ade_mean_distances, dim=1)
                ade_min_distances = distances[
                    torch.arange(0, len(ade_index_min), device=output.device),
                    ade_index_min
                ]
                ade_48s += torch.sum(torch.mean(ade_min_distances, dim=1))

                samples += distances.shape[0]

            # ---------------------------------------------------------
            # Save per-sample predictions for later visualization/analysis.
            # In P-TTA evaluation:
            #   past_rel / past_abs       = corrupted past actually seen by the model
            #   past_rel_clean / past_abs_clean = original clean past before corruption
            #   valid_mask                = which observed points are available
            #   tta_loss_*                = test-time adaptation diagnostic stats
            # ---------------------------------------------------------
            if save_enabled:
                batch_size = trajectory.shape[0]
                batch_indices = torch.arange(0, batch_size, device=output.device)

                # Relative coordinates.
                # past_rel is the corrupted input actually fed into PPT/P-TTA.
                past_rel = x_shift.detach().cpu().numpy()           # [B, 8, 2]
                past_rel_clean = x.detach().cpu().numpy()           # [B, 8, 2]
                future_rel = y.detach().cpu().numpy()               # [B, 12, 2]
                pred20_rel = output.detach().cpu().numpy()          # [B, 20, 12, 2]
                valid_mask_np = valid_mask.detach().cpu().numpy()   # [B, 8]

                best_ade_rel = output[
                    batch_indices,
                    ade_index_min
                ].detach().cpu().numpy()                            # [B, 12, 2]

                best_fde_rel = output[
                    batch_indices,
                    fde_index_min
                ].detach().cpu().numpy()                            # [B, 12, 2]

                # Absolute/world coordinates.
                # past_abs is the corrupted input actually fed into PPT/P-TTA.
                past_abs = abs_past_shift.detach().cpu().numpy()     # [B, 8, 2]
                past_abs_clean = abs_past.detach().cpu().numpy()     # [B, 8, 2]
                future_abs = trajectory[:, self.config.past_len:, :].detach().cpu().numpy()

                pred20_abs = (
                    output + initial_pose.unsqueeze(1)
                ).detach().cpu().numpy()                            # [B, 20, 12, 2]

                best_ade_abs = (
                    output[batch_indices, ade_index_min] + initial_pose
                ).detach().cpu().numpy()                            # [B, 12, 2]

                best_fde_abs = (
                    output[batch_indices, fde_index_min] + initial_pose
                ).detach().cpu().numpy()                            # [B, 12, 2]

                sample_ids = np.arange(sample_offset, sample_offset + batch_size)
                scenes = np.array([self.config.data_scene] * batch_size)

                save_past_rel.append(past_rel)
                save_future_rel.append(future_rel)
                save_pred20_rel.append(pred20_rel)
                save_best_ade_rel.append(best_ade_rel)
                save_best_fde_rel.append(best_fde_rel)

                save_past_abs.append(past_abs)
                save_future_abs.append(future_abs)
                save_pred20_abs.append(pred20_abs)
                save_best_ade_abs.append(best_ade_abs)
                save_best_fde_abs.append(best_fde_abs)

                save_past_rel_clean.append(past_rel_clean)
                save_past_abs_clean.append(past_abs_clean)
                save_valid_mask.append(valid_mask_np)

                save_tta_loss_first.append(
                    np.array([tta_stats["tta_loss_first"]] * batch_size)
                )
                save_tta_loss_last.append(
                    np.array([tta_stats["tta_loss_last"]] * batch_size)
                )
                save_tta_valid_next_ratio.append(
                    np.array([tta_stats["valid_next_ratio"]] * batch_size)
                )

                save_ade_index.append(ade_index_min.detach().cpu().numpy())
                save_fde_index.append(fde_index_min.detach().cpu().numpy())
                save_sample_id.append(sample_ids)
                save_scene.append(scenes)

                sample_offset += batch_size

            # ---------------------------------------------------------
            # Original PPT visualization logic
            # ---------------------------------------------------------
            if self.config.vis:
                with torch.no_grad():
                    if self.config.dataset_name == 'eth':
                        # Transform trajectories from world coordinate system to pixel/image space
                        trajectory_ = np.concatenate(
                            (
                                trajectory.cpu().numpy(),
                                np.ones((trajectory.size(0), trajectory.size(1), 1))
                            ),
                            -1
                        )
                        trajectory_ = np.matmul(
                            np.linalg.inv(self.homo_mat[self.config.data_scene]),
                            trajectory_.transpose(0, 2, 1)
                        ).transpose(0, 2, 1)
                        trajectory_ /= trajectory_[:, :, -1:]

                        output_ = output + initial_pose.unsqueeze(1)
                        output_ = np.concatenate(
                            (
                                output_.cpu().numpy(),
                                np.ones((output_.size(0), output_.size(1), output_.size(2), 1))
                            ),
                            -1
                        )
                        output_ = np.matmul(
                            np.linalg.inv(self.homo_mat[self.config.data_scene]),
                            output_.transpose(0, 1, 3, 2)
                        ).transpose(0, 1, 3, 2)
                        output_ /= output_[:, :, :, -1:]

                        if self.config.data_scene == 'eth' or self.config.data_scene == 'ucy':
                            trajectory_ = np.concatenate(
                                (trajectory_[:, :, 1:2], trajectory_[:, :, :1]),
                                -1
                            )
                            output_ = np.concatenate(
                                (output_[:, :, :, 1:2], output_[:, :, :, :1]),
                                -1
                            )
                        else:
                            trajectory_ = trajectory_[:, :, :2]
                            output_ = output_[:, :, :, :2]

                    elif self.config.dataset_name == 'sdd':
                        trajectory_ = trajectory.cpu().numpy()
                        output_ = output.cpu().numpy()

                    past_gt.append(trajectory_[:, :self.config.past_len])
                    fut_gt.append(trajectory_[:, self.config.past_len:])
                    fut_pred_20.append(output_)
                    fut_pred_best.append(
                        output_[np.arange(0, len(ade_index_min)), ade_index_min.cpu().numpy()]
                    )

        # ---------------------------------------------------------
        # Final metrics
        # ---------------------------------------------------------
        dict_metrics['ade_48s'] = ade_48s / samples
        dict_metrics['fde_48s'] = fde_48s / samples

        # ---------------------------------------------------------
        # Write per-sample predictions to a compressed npz file
        # ---------------------------------------------------------
        if save_enabled:
            if len(save_past_rel) == 0:
                print("[Warning] save_preds=True but no predictions were collected.")
            else:
                os.makedirs(os.path.join("results", "predictions"), exist_ok=True)

                pred_tag = getattr(self.config, "pred_tag", "ptta")
                save_path = os.path.join(
                    "results",
                    "predictions",
                    f"{self.config.data_scene}_{pred_tag}.npz"
                )

                past_rel_all = np.concatenate(save_past_rel, axis=0)
                future_rel_all = np.concatenate(save_future_rel, axis=0)
                pred20_rel_all = np.concatenate(save_pred20_rel, axis=0)
                best_ade_rel_all = np.concatenate(save_best_ade_rel, axis=0)
                best_fde_rel_all = np.concatenate(save_best_fde_rel, axis=0)

                past_abs_all = np.concatenate(save_past_abs, axis=0)
                future_abs_all = np.concatenate(save_future_abs, axis=0)
                pred20_abs_all = np.concatenate(save_pred20_abs, axis=0)
                best_ade_abs_all = np.concatenate(save_best_ade_abs, axis=0)
                best_fde_abs_all = np.concatenate(save_best_fde_abs, axis=0)

                past_rel_clean_all = np.concatenate(save_past_rel_clean, axis=0)
                past_abs_clean_all = np.concatenate(save_past_abs_clean, axis=0)
                valid_mask_all = np.concatenate(save_valid_mask, axis=0)

                tta_loss_first_all = np.concatenate(save_tta_loss_first, axis=0)
                tta_loss_last_all = np.concatenate(save_tta_loss_last, axis=0)
                tta_valid_next_ratio_all = np.concatenate(save_tta_valid_next_ratio, axis=0)

                ade_index_all = np.concatenate(save_ade_index, axis=0)
                fde_index_all = np.concatenate(save_fde_index, axis=0)
                sample_id_all = np.concatenate(save_sample_id, axis=0)
                scene_all = np.concatenate(save_scene, axis=0)

                np.savez_compressed(
                    save_path,

                    # Corrupted past actually seen by PPT/P-TTA.
                    past_rel=past_rel_all,
                    past_abs=past_abs_all,

                    # Clean original past before corruption.
                    past_rel_clean=past_rel_clean_all,
                    past_abs_clean=past_abs_clean_all,
                    valid_mask=valid_mask_all,

                    # Clean future GT and predictions from corrupted input.
                    future_rel=future_rel_all,
                    future_abs=future_abs_all,
                    pred20_rel=pred20_rel_all,
                    pred20_abs=pred20_abs_all,
                    best_ade_rel=best_ade_rel_all,
                    best_ade_abs=best_ade_abs_all,
                    best_fde_rel=best_fde_rel_all,
                    best_fde_abs=best_fde_abs_all,

                    ade_index=ade_index_all,
                    fde_index=fde_index_all,
                    sample_id=sample_id_all,
                    scene=scene_all,

                    dataset_name=np.array(self.config.dataset_name),
                    data_scene=np.array(self.config.data_scene),
                    model_path=np.array(self.config.model_Pretrain),
                    past_len=np.array(self.config.past_len),
                    future_len=np.array(self.config.future_len),
                    ade_48s=np.array(self._to_float(dict_metrics["ade_48s"])),
                    fde_48s=np.array(self._to_float(dict_metrics["fde_48s"])),
                    info=np.array(getattr(self.config, "info", "")),

                    # Robustness metadata.
                    shift_type=np.array(getattr(self.config, "shift_type", "clean")),
                    missing_rate=np.array(getattr(self.config, "missing_rate", 0.0)),
                    missing_pattern=np.array(getattr(self.config, "missing_pattern", "random")),
                    missing_fill=np.array(getattr(self.config, "missing_fill", "last")),
                    noise_alpha=np.array(getattr(self.config, "noise_alpha", 0.0)),
                    drop_num=np.array(getattr(self.config, "drop_num", 0)),
                    robust_tag=np.array(getattr(self.config, "robust_tag", "robust")),

                    # P-TTA metadata.
                    tta_loss_first=tta_loss_first_all,
                    tta_loss_last=tta_loss_last_all,
                    tta_valid_next_ratio=tta_valid_next_ratio_all,
                    tta_steps=np.array(getattr(self.config, "tta_steps", 1)),
                    tta_lr=np.array(getattr(self.config, "tta_lr", 1e-4)),
                    tta_loss_type=np.array(getattr(self.config, "tta_loss_type", "smooth_l1")),
                    tta_update_head=np.array(getattr(self.config, "tta_update_head", False)),
                    tta_require_source_valid=np.array(getattr(self.config, "tta_require_source_valid", False)),
                    ptta_init_checkpoint=np.array(getattr(self.config, "ptta_init_checkpoint", "")),
                    ptta_tag=np.array(getattr(self.config, "ptta_tag", "ptta")),
                )

                print(f"[Saved predictions] {save_path}")
                print(f"  past_abs                 : {past_abs_all.shape}  # corrupted input")
                print(f"  past_abs_clean           : {past_abs_clean_all.shape}  # original clean input")
                print(f"  valid_mask               : {valid_mask_all.shape}")
                print(f"  future_abs               : {future_abs_all.shape}")
                print(f"  pred20_abs               : {pred20_abs_all.shape}")
                print(f"  best_ade_abs             : {best_ade_abs_all.shape}")
                print(f"  best_fde_abs             : {best_fde_abs_all.shape}")
                print(f"  tta_loss_first           : {tta_loss_first_all.shape}")
                print(f"  tta_loss_last            : {tta_loss_last_all.shape}")
                print(f"  tta_valid_next_ratio     : {tta_valid_next_ratio_all.shape}")

        # ---------------------------------------------------------
        # Original PPT visualization output
        # ---------------------------------------------------------
        if self.config.vis:
            past_gt = np.concatenate(past_gt, 0)
            fut_gt = np.concatenate(fut_gt, 0)
            fut_pred_20 = np.concatenate(fut_pred_20, 0)
            fut_pred_best = np.concatenate(fut_pred_best, 0)

            if self.config.dataset_name == 'eth':
                vis_ETH(
                    self.config.data_scene,
                    past_gt,
                    fut_gt,
                    fut_pred_20,
                    fut_pred_best
                )
            # else:
            #     vis_SDD(past_gt, fut_gt, fut_pred_20, fut_pred_best)

        return dict_metrics
