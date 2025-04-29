import os
import sys
import argparse
import torch
import wandb

from src.utils import load_config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# source /datascope/slurm/miniconda3/bin/activate viska-torch-2
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['WANDB_ENTITY'] = 'viskawei-johns-hopkins-university'
# os.environ['CUDA_VISIBLE_DEVICES']='1,2,3,4,5,6,7'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


from src.blindspot import Experiment

ckpt=None
# ckpt = '/home/swei20/SirenSpec/src/artifacts/model-ypx736v3:v0/model.ckpt'

def parse_args():
    parser = argparse.ArgumentParser(description='blindspot experiment')
    parser.add_argument('-f', '--config', type=str, help='config file')
    parser.add_argument('-w', '--wandb', type=int, help='use wandb logging', default=0)
    parser.add_argument('-g', '--gpu', type=int, help='gpu number', default=torch.cuda.device_count())
    parser.add_argument('--debug', type=int, help='debug mode', default=0)
    parser.add_argument('--ckpt', type=str, help='path to checkpoint file', default=ckpt)
    return parser.parse_args()

def main(args: argparse.Namespace) -> None:
    config = load_config(args.config or 'configs/blindspot.yaml')
    config['train']['gpus'] = args.gpu
    config['train']['debug'] = args.debug
    Experiment(config, use_wandb=args.wandb, sweep=False, ckpt_path=args.ckpt).run()

if __name__ == "__main__":
    args = parse_args()
    main(args)