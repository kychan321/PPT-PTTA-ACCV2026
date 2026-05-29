import argparse
import logging
import os
import random

import numpy as np
import torch

from trainer import trainer_ptta_init as trainer_ppt


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in ("yes", "true", "t", "1", "y"):
        return True
    if v in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def seed_torch(seed=1666):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_config():
    parser = argparse.ArgumentParser(
        description="Offline initialization for PPT-based P-TTA adapter/pretext head"
    )

    # Basic runtime
    parser.add_argument("--cuda", type=str2bool, default=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1666)
    parser.add_argument("--info", type=str, default="ptta_init")

    # Dataset
    parser.add_argument("--dataset_name", type=str, default="eth")
    parser.add_argument("--data_scene", type=str, default="eth")
    parser.add_argument("--train_b_size", type=int, default=512)
    parser.add_argument("--test_b_size", type=int, default=4096)
    parser.add_argument("--time_thresh", type=int, default=0)
    parser.add_argument("--dist_thresh", type=int, default=50)

    # Trajectory length
    parser.add_argument("--past_len", type=int, default=8)
    parser.add_argument("--future_len", type=int, default=12)

    # PPT transformer config
    # Keep these aligned with PPT pretrained model config.
    parser.add_argument("--dim_embedding_key", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--T", type=int, default=8)
    parser.add_argument("--vocab_size", type=int, default=2)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--embd_pdrop", type=float, default=0.1)
    parser.add_argument("--resid_pdrop", type=float, default=0.1)
    parser.add_argument("--attn_pdrop", type=float, default=0.1)

    # Required pretrained PPT checkpoint
    parser.add_argument(
        "--model_Pretrain",
        type=str,
        required=True,
        help="Path to scene-specific pretrained PPT checkpoint, e.g. model_univ_res.ckpt"
    )

    # P-TTA module config
    parser.add_argument("--ptta_adapter_bottleneck", type=int, default=32)
    parser.add_argument("--ptta_adapter_dropout", type=float, default=0.0)
    parser.add_argument("--ptta_pretext_hidden_dim", type=int, default=64)
    parser.add_argument("--ptta_pretext_dropout", type=float, default=0.0)
    parser.add_argument("--ptta_adapt_destination", type=str2bool, default=False)

    # Offline init training config
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--loss_type", type=str, default="smooth_l1",
                        choices=["smooth_l1", "l1", "l2"])
    parser.add_argument("--require_source_valid", type=str2bool, default=False)

    # Save interval
    parser.add_argument("--save_every", type=int, default=5)

    # Force mode ALL because model_test_trajectory_ptta expects final PPT checkpoint.
    parser.add_argument("--mode", type=str, default="ALL")

    return parser.parse_args()


def main(config):
    seed_torch(config.seed)

    if config.cuda and not torch.cuda.is_available():
        print("[Warning] CUDA requested but not available. Falling back to CPU.")
        config.cuda = False

    print(config)

    trainer = trainer_ppt.Trainer(config)
    trainer.fit()


if __name__ == "__main__":
    config = parse_config()
    main(config)