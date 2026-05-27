import os
import datetime
import csv

import numpy as np
import torch
import torch.nn as nn
from models.model_test_trajectory import Final_Model
from torch.utils.data import DataLoader

from dataset_loader import *
from vis_ethucy import *

torch.set_num_threads(5)


class Trainer:
    def __init__(self, config):

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

        if torch.cuda.is_available():
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

        # Load pretrained PPT model
        self.model_Pretrain = torch.load(
            config.model_Pretrain,
            map_location=torch.device('cpu')
        ).cuda()

        self.model = Final_Model(config, self.model_Pretrain)

        if config.cuda:
            self.model = self.model.cuda()

        self.start_epoch = 0
        self.config = config
        self.device = torch.device('cuda') if config.cuda else torch.device('cpu')

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

    def fit(self):

        dict_metrics_test = self.evaluate(self.test_dataset)

        ade = self._to_float(dict_metrics_test["ade_48s"])
        fde = self._to_float(dict_metrics_test["fde_48s"])

        print('[Clean Eval]')
        print('ADE_48s: {:.6f} | FDE_48s: {:.6f}'.format(ade, fde))
        print('-' * 100)

        # ---------------------------------------------------------
        # Save clean evaluation result to CSV
        # ---------------------------------------------------------
        os.makedirs("results/clean", exist_ok=True)

        csv_path = os.path.join("results", "clean", f"{self.config.data_scene}.csv")
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
                    "info"
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
                self.config.info
            ])

        print(f"[Saved CSV] {csv_path}")

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

        save_ade_index = []
        save_fde_index = []
        save_sample_id = []
        save_scene = []

        sample_offset = 0

        with torch.no_grad():
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
                # Model prediction
                # output shape: [B, 20, 12, 2]
                # ---------------------------------------------------------
                output = self.model(x, abs_past, seq_start_end, initial_pose)
                output = output.data

                # ---------------------------------------------------------
                # Compute ADE/FDE using best-of-20 protocol
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
                # Save per-sample predictions for later visualization/analysis
                # ---------------------------------------------------------
                if save_enabled:
                    batch_size = trajectory.shape[0]
                    batch_indices = torch.arange(0, batch_size, device=output.device)

                    # Relative coordinates:
                    # These are normalized by the last observed point x8.
                    past_rel = x.detach().cpu().numpy()                 # [B, 8, 2]
                    future_rel = y.detach().cpu().numpy()               # [B, 12, 2]
                    pred20_rel = output.detach().cpu().numpy()          # [B, 20, 12, 2]

                    best_ade_rel = output[
                        batch_indices,
                        ade_index_min
                    ].detach().cpu().numpy()                            # [B, 12, 2]

                    best_fde_rel = output[
                        batch_indices,
                        fde_index_min
                    ].detach().cpu().numpy()                            # [B, 12, 2]

                    # Absolute/world coordinates:
                    # output is relative to the last observed point, so add initial_pose back.
                    past_abs = trajectory[:, :self.config.past_len, :].detach().cpu().numpy()
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

                    save_ade_index.append(ade_index_min.detach().cpu().numpy())
                    save_fde_index.append(fde_index_min.detach().cpu().numpy())
                    save_sample_id.append(sample_ids)
                    save_scene.append(scenes)

                    sample_offset += batch_size

                # ---------------------------------------------------------
                # Original PPT visualization logic
                # ---------------------------------------------------------
                if self.config.vis:
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

                    pred_tag = getattr(self.config, "pred_tag", "clean")
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

                    ade_index_all = np.concatenate(save_ade_index, axis=0)
                    fde_index_all = np.concatenate(save_fde_index, axis=0)
                    sample_id_all = np.concatenate(save_sample_id, axis=0)
                    scene_all = np.concatenate(save_scene, axis=0)

                    np.savez_compressed(
                        save_path,

                        past_rel=past_rel_all,
                        future_rel=future_rel_all,
                        pred20_rel=pred20_rel_all,
                        best_ade_rel=best_ade_rel_all,
                        best_fde_rel=best_fde_rel_all,

                        past_abs=past_abs_all,
                        future_abs=future_abs_all,
                        pred20_abs=pred20_abs_all,
                        best_ade_abs=best_ade_abs_all,
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
                    )

                    print(f"[Saved predictions] {save_path}")
                    print(f"  past_abs     : {past_abs_all.shape}")
                    print(f"  future_abs   : {future_abs_all.shape}")
                    print(f"  pred20_abs   : {pred20_abs_all.shape}")
                    print(f"  best_ade_abs : {best_ade_abs_all.shape}")
                    print(f"  best_fde_abs : {best_fde_abs_all.shape}")

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