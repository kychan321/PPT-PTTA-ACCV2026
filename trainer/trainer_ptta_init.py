import os
import csv
import datetime
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_loader import *
from models.model_test_trajectory_ptta import Final_Model

torch.set_num_threads(5)


class Trainer:
    def __init__(self, config):
        self.config = config

        # ---------------------------------------------------------
        # Output folder
        # ---------------------------------------------------------
        self.name_test = str(datetime.datetime.now())[:10]
        self.folder_test = os.path.join(
            "training",
            "ptta_init",
            f"{self.name_test}_{config.info}"
        )

        os.makedirs(self.folder_test, exist_ok=True)
        self.folder_test = self.folder_test + "/"

        # ---------------------------------------------------------
        # Device
        # ---------------------------------------------------------
        if torch.cuda.is_available() and config.cuda:
            torch.cuda.set_device(config.gpu)

        self.device = torch.device("cuda") if config.cuda else torch.device("cpu")

        # ---------------------------------------------------------
        # Dataset
        # ---------------------------------------------------------
        print("Preprocess data")

        if config.dataset_name == "sdd":
            data_folder = "data"
            train_dataset = SocialDataset(
                data_folder,
                set_name="train",
                b_size=512,
                t_tresh=0,
                d_tresh=100,
                scene="sdd"
            )
        elif config.dataset_name == "eth":
            data_folder = "data/ETH_UCY"
            train_dataset = SocialDataset(
                data_folder,
                set_name="train",
                b_size=256,
                t_tresh=0,
                d_tresh=50,
                scene=config.data_scene
            )
        else:
            raise ValueError(f"Unsupported dataset_name: {config.dataset_name}")

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            collate_fn=socialtraj_collate,
            shuffle=True
        )

        print("Loaded train data!")

        # ---------------------------------------------------------
        # Load pretrained PPT and build P-TTA model
        # ---------------------------------------------------------
        config.mode = "ALL"

        print(f"Loading pretrained PPT checkpoint: {config.model_Pretrain}")
        pretrained_model = torch.load(
            config.model_Pretrain,
            map_location=torch.device("cpu")
        )

        if config.cuda:
            pretrained_model = pretrained_model.cuda()

        self.model = Final_Model(config, pretrained_model)

        if config.cuda:
            self.model = self.model.cuda()

        # Freeze PPT backbone and train only P-TTA modules.
        self.model.freeze_ppt_backbone()

        self.optimizer = torch.optim.Adam(
            self.model.get_tta_parameters(update_head=True),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        self.max_epochs = config.max_epochs

        # ---------------------------------------------------------
        # Logging
        # ---------------------------------------------------------
        self.logger = logging.getLogger(f"ptta_init_{config.info}")
        self.logger.setLevel(level=logging.DEBUG)
        self.logger.handlers = []

        formatter = logging.Formatter(
            "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
        )

        file_handler = logging.FileHandler(os.path.join(self.folder_test, "train.log"))
        file_handler.setLevel(level=logging.INFO)
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

        # CSV log
        self.csv_path = os.path.join(self.folder_test, "loss_log.csv")
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss",
                "avg_valid_next_ratio",
                "learning_rate",
                "data_scene",
                "model_pretrain"
            ])

        self.best_train_loss = float("inf")

        self.logger.info("Initialized P-TTA offline trainer.")
        self.print_trainable_parameters()

    # ---------------------------------------------------------
    # Utility
    # ---------------------------------------------------------
    def print_trainable_parameters(self):
        total_num = sum(p.numel() for p in self.model.parameters())
        trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.logger.info(f"Trainable/Total parameters: {trainable_num}/{total_num}")

        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self.logger.info(f"  trainable: {name} | shape={tuple(p.shape)}")

    def set_ptta_train_mode(self):
        """
        Keep frozen PPT backbone deterministic and train only adapter/head.

        Important:
            Calling self.model.train() would also enable dropout inside the frozen PPT.
            Instead, keep the full model in eval mode and explicitly set only
            P-TTA modules to train mode.
        """
        self.model.eval()
        self.model.past_adapter.train()
        self.model.pretext_head.train()

    def save_ptta_checkpoint(self, filename, epoch, train_loss):
        save_path = os.path.join(self.folder_test, filename)

        state = self.model.get_ptta_state()
        state.update({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "data_scene": self.config.data_scene,
            "dataset_name": self.config.dataset_name,
            "model_Pretrain": self.config.model_Pretrain,
            "past_len": self.config.past_len,
            "future_len": self.config.future_len,
            "n_embd": self.config.n_embd,
            "vocab_size": self.config.vocab_size,
            "ptta_adapter_bottleneck": self.config.ptta_adapter_bottleneck,
            "ptta_pretext_hidden_dim": self.config.ptta_pretext_hidden_dim,
            "loss_type": self.config.loss_type,
            "require_source_valid": self.config.require_source_valid,
        })

        torch.save(state, save_path)
        self.logger.info(f"[Saved] {save_path}")

    # ---------------------------------------------------------
    # Main training loop
    # ---------------------------------------------------------
    def fit(self):
        self.logger.info("Start offline initialization for P-TTA adapter/pretext head.")
        self.logger.info(f"Output folder: {self.folder_test}")

        for epoch in range(self.max_epochs):
            train_loss, avg_valid_ratio = self._train_single_epoch(epoch)

            self.logger.info(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.6f} | "
                f"avg_valid_next_ratio={avg_valid_ratio:.4f}"
            )

            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    train_loss,
                    avg_valid_ratio,
                    self.config.learning_rate,
                    self.config.data_scene,
                    self.config.model_Pretrain
                ])

            # Save best checkpoint based on train self-supervised loss only.
            # We intentionally do not use test ADE/FDE for model selection here.
            if train_loss < self.best_train_loss:
                self.best_train_loss = train_loss
                self.save_ptta_checkpoint(
                    filename="ptta_init_state_best.pt",
                    epoch=epoch,
                    train_loss=train_loss
                )

            if self.config.save_every > 0 and (epoch + 1) % self.config.save_every == 0:
                self.save_ptta_checkpoint(
                    filename=f"ptta_init_state_epoch_{epoch + 1}.pt",
                    epoch=epoch,
                    train_loss=train_loss
                )

        self.save_ptta_checkpoint(
            filename="ptta_init_state_last.pt",
            epoch=self.max_epochs - 1,
            train_loss=train_loss
        )

        self.logger.info("Finished P-TTA offline initialization.")

    def _train_single_epoch(self, epoch):
        self.set_ptta_train_mode()

        total_loss = 0.0
        total_valid_ratio = 0.0
        count = 0

        for _, (trajectory, mask, initial_pos, seq_start_end) in enumerate(self.train_loader):
            trajectory = torch.FloatTensor(trajectory).to(self.device)

            # Same normalization as PPT:
            # x8 is the anchor point.
            traj_norm = trajectory - trajectory[:, self.config.past_len - 1:self.config.past_len, :]
            x = traj_norm[:, :self.config.past_len, :]  # [B, 8, 2]

            # In offline init, train split is clean.
            # All observed past points are valid.
            valid_mask = torch.ones(
                x.shape[:2],
                dtype=torch.bool,
                device=self.device
            )

            self.optimizer.zero_grad()

            loss, stats = self.model.pretext_next_loss(
                past=x,
                valid_mask=valid_mask,
                loss_type=self.config.loss_type,
                require_source_valid=self.config.require_source_valid,
                return_stats=True
            )

            loss.backward()

            if self.config.grad_clip is not None and self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.get_tta_parameters(update_head=True),
                    self.config.grad_clip,
                    norm_type=2
                )

            self.optimizer.step()

            total_loss += float(loss.detach().cpu().item())
            total_valid_ratio += stats["valid_next_ratio"]
            count += 1

        avg_loss = total_loss / max(count, 1)
        avg_valid_ratio = total_valid_ratio / max(count, 1)

        return avg_loss, avg_valid_ratio