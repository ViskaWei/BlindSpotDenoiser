import os
import sys
import argparse
import importlib.util
from pathlib import Path

import torch
import wandb

from src.utils import load_config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# source /datascope/slurm/miniconda3/bin/activate viska-torch-2
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['WANDB_ENTITY'] = 'viskawei-johns-hopkins-university'
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
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
    parser.add_argument('--ckpt', type=str, help='path to checkpoint file (Lightning resume: weights+optimizer+epoch)', default=ckpt)
    parser.add_argument('--init-from', dest='init_from', type=str, help='path to ckpt for weights-only init (fresh optimizer, epoch=0)', default=None)
    parser.add_argument('--preflight-id', dest='preflight_id', type=str,
                        help='Registered preflight manifest id (Layer C gate). '
                             'If omitted, PREFLIGHT_ID env var is used; '
                             'PREFLIGHT_TIER in {0,1} waives the gate.',
                        default=None)
    return parser.parse_args()

def main(args: argparse.Namespace) -> None:
    # ── Preflight gate (Layer C runtime token) ────────────────────────
    # Resource-boundary enforcement: any bash / ssh / tmux / nohup bypass
    # of the external hook is caught here before GPU/torch init proceeds.
    try:
        _preflight_path = (
            Path(__file__).resolve().parents[3]
            / "shared" / "skills" / "tools" / "preflight" / "require_preflight.py"
        )
        if _preflight_path.exists():
            _spec = importlib.util.spec_from_file_location(
                "require_preflight", _preflight_path
            )
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            _mod.require_preflight_or_die(args.config, args.preflight_id)
    except SystemExit:
        raise
    except Exception as _exc:
        # Fail-open on gate module itself crashing so we never brick the repo
        # due to a bug in this file. Bash hook remains the outer layer.
        print(f"[preflight] WARNING: gate check failed to execute: {_exc}",
              file=sys.stderr)

    config = load_config(args.config or 'configs/mu_only_baseline_mag205_225.yaml')
    config['train']['gpus'] = args.gpu
    config['train']['debug'] = args.debug
    Experiment(config, use_wandb=args.wandb, sweep=False, ckpt_path=args.ckpt, init_from=args.init_from).run()

if __name__ == "__main__":
    args = parse_args()
    main(args)