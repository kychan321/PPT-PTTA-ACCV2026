import argparse
from trainer import test_final_trajectory_ptta as trainer_ppt

import numpy as np
import random
import torch


def prepare_seed(rand_seed):
	np.random.seed(rand_seed)
	random.seed(rand_seed)
	torch.manual_seed(rand_seed)
	torch.cuda.manual_seed_all(rand_seed)

    
def parse_config():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument("--cuda", default=True)

    parser.add_argument("--past_len", type=int, default=8, help="length of past (in timesteps)")
    parser.add_argument("--future_len", type=int, default=12, help="length of future (in timesteps)")
    parser.add_argument("--dim_embedding_key", type=int, default=24)
    parser.add_argument("--data_scale", type=float, default=1)
    parser.add_argument("--data_scale_old", type=float, default=1.86)
    parser.add_argument("--train_b_size", type=int, default=512)
    parser.add_argument("--test_b_size", type=int, default=4096)
    parser.add_argument("--time_thresh", type=int, default=0)
    parser.add_argument("--dist_thresh", type=int, default=100)
    
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument("--model_Pretrain", default='./training/...')

    parser.add_argument("--save_preds", action="store_true",
                    help="Save per-sample past, future GT, and predictions to results/predictions.")
    parser.add_argument("--pred_tag", type=str, default="clean",
                    help="Tag used for saved prediction npz filename.")
    
    parser.add_argument("--shift_type", type=str, default="clean",
                    choices=["clean", "missing", "noise", "frame_drop"],
                    help="Type of test-time observation corruption.")
    parser.add_argument("--missing_rate", type=float, default=0.0,
                    help="Missing ratio applied to observed past x1~x7. x8 is kept.")
    parser.add_argument("--missing_pattern", type=str, default="random",
                    choices=["random", "consecutive"],
                    help="Pattern for missing observation.")
    parser.add_argument("--missing_fill", type=str, default="last",
                    choices=["zero", "last", "linear"],
                    help="How to fill missing observed points before feeding PPT.")
    parser.add_argument("--noise_alpha", type=float, default=0.0,
                    help="Gaussian noise scale relative to average observed step displacement.")
    parser.add_argument("--drop_num", type=int, default=0,
                    help="Number of observed frames to drop among x1~x7. x8 is kept.")
    parser.add_argument("--robust_tag", type=str, default="robust",
                    help="Tag saved in robust CSV.")

    parser.add_argument("--reproduce", action="store_true")
    parser.add_argument("--vis", action="store_true")

    parser.add_argument("--ptta_init_checkpoint", type=str, default="",
                    help="Path to offline-initialized P-TTA adapter/head checkpoint.")

    parser.add_argument("--tta_steps", type=int, default=1,
                    help="Number of test-time adaptation steps per batch/sample.")

    parser.add_argument("--tta_lr", type=float, default=1e-4,
                    help="Learning rate for test-time adapter update.")

    parser.add_argument("--tta_grad_clip", type=float, default=1.0,
                    help="Gradient clipping norm for test-time update.")

    parser.add_argument("--tta_loss_type", type=str, default="smooth_l1",
                    choices=["smooth_l1", "l1", "l2"],
                    help="Loss type for test-time self-supervised next-position loss.")

    parser.add_argument("--tta_update_head", action="store_true",
                    help="If set, update both adapter and pretext head at test time. Default updates adapter only.")

    parser.add_argument("--tta_require_source_valid", action="store_true",
                    help="Use next-position targets only when both source and target are valid.")

    parser.add_argument("--ptta_tag", type=str, default="ptta",
                    help="Tag saved in P-TTA CSV.")

    parser.add_argument("--dataset_file", default="SDD", help="dataset file")
    parser.add_argument("--dataset_name", default="sdd", help="dataset file")
    parser.add_argument("--data_scene", default="eth", help="dataset file")
    parser.add_argument("--info", type=str, default='', help='Name of training. '
                                                             'It will be used in tensorboard log and test folder')

    return parser.parse_args()


def main(config):
    if config.reproduce:
        config.model_Pretrain = './training/Pretrained_Models/SDD/model_ALL'
        print(config.model_Pretrain)
        t = trainer_ppt.Trainer(config)
        t.fit()
    else:
        print(config.model_Pretrain)
        t = trainer_ppt.Trainer(config)
        t.fit()


if __name__ == "__main__":
    config = parse_config()
    main(config)
