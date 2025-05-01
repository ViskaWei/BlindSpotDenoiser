import os
from typing import Optional, Dict, Any
import wandb
from src.blindspot import Experiment

os.environ['WANDB_ENTITY'] = 'viskawei-johns-hopkins-university'
wandb_dir = "/datascope/subaru/user/swei20/wandb"

def main(config: Optional[Dict[str, Any]] = None) -> None:
    with wandb.init(dir=wandb_dir, save_code=False, settings=wandb.Settings(_disable_stats=True)) as run:
        config = wandb.config

        gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A')
        print(f"Running on GPU: {gpu_id}")

        e = Experiment(config, use_wandb=True, sweep=True)
        run.name = e.lightning_module.model.name
        e.run()
        
if __name__ == "__main__":
    main()
    
    
    