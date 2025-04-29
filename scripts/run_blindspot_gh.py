import os
import sys
import argparse
import torch
import wandb

from src.utils import load_config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['WANDB_ENTITY'] = 'viskawei-johns-hopkins-university'
# os.environ['CUDA_VISIBLE_DEVICES']='0'


from src.blindspot import Experiment

ckpt=None
# ckpt= '/datascope/subaru/user/swei20/wandb/blindspot-best/bc1x8icw/checkpoints/ep700.ckpt'
# ckpt= '/datascope/subaru/user/swei20/blindspot/checkpoints/epoch=499-snr_valid=0-v2.ckpt'
# ckpt = '/home/swei20/SirenSpec/blindspot-best/7voytghz/checkpoints/epoch=99-step=156300.ckpt'
# ckpt = '/home/swei20/SirenSpec/blindspot-best/ydsi6y16/checkpoints/epoch=414-step=648645.ckpt'
# ckpt='/home/swei20/SirenSpec/blindspot-best/blwswd9s/checkpoints/epoch=577-step=903414.ckpt'
# viskawei-johns-hopkins-university/blindspot-best/model-f5y0xzqg:v0
# ckpt='/datascope/subaru/user/swei20/wandb/blindspot-best/1nqedile/checkpoints/epoch=692-step=1083159.ckpt'
# ckpt = '/datascope/subaru/user/swei20/wandb/blindspot-best/f5y0xzqg/checkpoints/epoch=199-step=156400.ckpt'
# ckpt='/home/swei20/SirenSpec/checkpoints/gawuh4kw.ckpt'
# ckpt = '/datascope/subaru/user/swei20/wandb/blindspot-best/ycbsq4zn/checkpoints/epoch=199-step=156400.ckpt'
# ckpt = '/datascope/subaru/user/swei20/wandb/blindspot-best/4httl4f2/checkpoints/epoch=99-step=78200.ckpt'
# ckpt = '/datascope/subaru/user/swei20/wandb/blindspot-best/bsg3h4ag/checkpoints/epoch=99-step=78200.ckpt'

def parse_args():
    parser = argparse.ArgumentParser(description='blindspot experiment')
    parser.add_argument('-f', '--config', type=str, help='config file')
    parser.add_argument('-w', '--wandb', type=int, help='use wandb logging', default=1)
    parser.add_argument('-g', '--gpu', type=int, help='gpu number', default=1)
    parser.add_argument('--debug', type=int, help='debug mode', default=0)
    parser.add_argument('--ckpt', type=str, help='path to checkpoint file', default=ckpt)
    return parser.parse_args()

def main(args: argparse.Namespace) -> None:
    config = load_config(args.config or 'configs/blindspot_gh.yaml')
    config['train']['gpus'] = args.gpu
    config['train']['debug'] = args.debug
    Experiment(config, use_wandb=args.wandb, sweep=False, ckpt_path=args.ckpt).run()

if __name__ == "__main__":
    args = parse_args()
    main(args)