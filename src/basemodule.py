import os
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import lightning as L
from torch.utils.data import Dataset, TensorDataset, DataLoader
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, List, Dict, Any, Type
# from src.utils import create_gif
from scipy import constants
#region DATA-----------------------------------------------------------
class Configurable:
    init_params = []
    config_section = None
    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        params = {}
        for base in cls.__mro__[::-1]:  # Reverse MRO to ensure correct parameter order
            if issubclass(base, Configurable) and base is not Configurable:
                if base.config_section:
                    section = config.get(base.config_section, {})
                    for param in base.init_params:
                        if param in section: params[param] = section[param]
        return cls(**params)

class BaseDataset(Configurable, Dataset, ABC):
    init_params = ['file_path', 'val_path', 'test_path', 'num_samples', 'num_test_samples', 'indices', 'root_dir']
    config_section = 'data'

    def __init__(self, file_path: str=None, num_samples: Optional[int] = None, test_path=None, num_test_samples: Optional[int] = None, val_path = None, indices: List[int] = None, root_dir: str = './results', **kwargs):
        super().__init__(**kwargs)
        print(file_path, num_samples, test_path, num_test_samples, val_path, indices, root_dir)
        self.file_path = file_path
        self.val_path = val_path if val_path is not None else file_path
        self.test_path = test_path if test_path is not None else file_path
        self.num_samples = num_samples if num_samples is not None else 1
        self.num_test_samples = num_test_samples if num_test_samples is not None else min(10000, self.num_samples)
        self.indices = indices if indices is not None else [0, 1]
        self.root_dir = root_dir
        self.test_data_dict = {}
        
    def prepare_data(self):
        """# called only on 1 GPU Prepare the data for training."""
        pass
    @abstractmethod
    def load_data(self, stage=None):
        """called on all GPU Load the data from the file."""
        pass
    @abstractmethod
    def __getitem__(self, idx: int):
        pass
    def __len__(self) -> int:
        return self.num_samples

class MaskMixin(Configurable):
    init_params = ['mask_ratio', 'mask_filler', 'mask', 'lvrg_num', 'lvrg_mask']
    config_section = 'mask'
    def __init__(self, mask_ratio: float = None, mask_filler: float = None, mask: List[int] = None, lvrg_num=None, lvrg_mask=None, **kwargs):
        super().__init__(**kwargs)
        self.mask_ratio = mask_ratio
        self.mask_filler = mask_filler
        self.mask = mask
        self.lvrg_num = lvrg_num
        self.lvrg_mask = None
        
    def fill_masked(self, tensor, filler=None):
        if filler is None: return tensor[..., self.mask]
        return tensor.masked_fill(~self.mask, filler)
    
    def create_quantile_mask(self, tensor, ratio=0.9):
        median = torch.median(tensor, dim=0).values
        print('median', median.mean())  #ratio: sigma,  0.4: 0.0098, 0.5: 0.01, 0.6: 0.0105, 0.8:0.0129, 0.85:0.0162(1.5) , 0.9: 0.0225, 1:0.048
        return median < torch.quantile(median, ratio)
    
    def create_lvrg_mask(self, wave, pdxs):
        wave_len = len(wave)
        mask = np.zeros(wave_len, dtype=bool)
        wdxs = np.digitize(pdxs, wave)
        for wdx in wdxs:
            start, end = max(0, wdx-25), min(wdx+25, wave_len)
            mask[start:end] = True
        return mask
        
        
        
class NoiseMixin(Configurable):
    init_params = ['noise_level', 'noise_max']
    config_section = 'noise'
    def __init__(self, noise_level: float = 1.0, noise_max: Optional[float] = None, **kwargs):
        super().__init__(**kwargs)
        self.noise_level = noise_level
        self.noise_max = noise_max    
        self.noisy = None
    @staticmethod
    def add_noise(flux: torch.Tensor, error: torch.Tensor, noise_level) -> torch.Tensor:
        return flux + torch.randn_like(flux) * error * noise_level
    def clamp_sigma(self, sigma):
        sigma = sigma.clamp(min=1e-6, max=self.noise_max)
        if self.noise_max is None: self.noise_max = round(sigma.max().item(), 2)
        self.sigma_rms = torch.sqrt(sigma.pow(2).mean(dim=-1)).mean()  #0.048 
        print('sigma_noise', self.sigma_rms)
        return sigma


class SingleSpectrumNoiseDataset(Dataset):
    def __init__(self, flux_0: torch.Tensor, error_0: torch.Tensor,
                 noise_level: float = 1.0, repeat: int = 1000, seed: int = 42):
        super().__init__()
        self.repeat = repeat
        self.noise_level = noise_level
        self.L = len(flux_0) 
        self.flux_0 = flux_0
        self.error_0 = error_0

        torch.manual_seed(seed)
        noise = torch.randn(repeat, self.L) * self.error_0 * self.noise_level
        self.noisy = self.flux_0 + noise
        print(self.noisy.shape, self.flux_0.shape, self.error_0.shape)

    def __len__(self):
        return self.repeat

    def __getitem__(self, idx):
        return self.noisy[idx], self.flux_0, self.error_0
        
class BaseSpecDataset(MaskMixin, NoiseMixin, BaseDataset):   
    def get_path_and_samples(self, stage):
        if stage == 'fit' or stage is None:
            return self.file_path, self.num_samples
        else:
            load_path = self.test_path if stage == 'test' else self.val_path
            return load_path, self.num_test_samples            
        
    def replace_nan_with_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        nan_mask = torch.isnan(tensor)
        mean_value = torch.median(tensor[~nan_mask])
        tensor[nan_mask] = mean_value
        return tensor

    def fill_nan_with_nearest(self, tensor):
        if torch.isnan(tensor[:, 0]).any():
            tensor[:, 0] = tensor[:, 1]  # Copy from second column
        if torch.isnan(tensor[:, -1]).any():
            tensor[:, -1] = tensor[:, -2]  # Copy from second last column
        return tensor

    def load_data(self, stage=None) -> None:
        load_path, num_samples = self.get_path_and_samples(stage)  
        print('loading data from', load_path, num_samples)          
        with h5py.File(load_path, 'r') as f:
            self.wave = torch.Tensor(f['spectrumdataset/wave'][()])
            self.flux = torch.Tensor(f['dataset/arrays/flux/value'][:num_samples])
            self.error = torch.Tensor(f['dataset/arrays/error/value'][:num_samples])

        self.flux = self.flux.clip(min=0.0)
        # self.error = self.error0.clip(min=1e-6, max=1.0)
        self.num_samples = self.flux.shape[0]
        self.num_pixels = len(self.wave)
        if self.error.isnan().any():
            # print(torch.isnan(self.error).sum())
            self.error = self.fill_nan_with_nearest(self.error)
        self.snr_no_mask =self.flux.norm(dim=-1) / self.error.norm(dim=-1)
        print(self.flux.shape, self.error.shape, self.wave.shape, self.num_samples, self.num_pixels)

          
    def load_snr(self, stage=None, load_df=False) -> None:
        load_path, num_samples = self.get_path_and_samples(stage)  
        df = pd.read_hdf(load_path)[:num_samples]          
        self.z = df['redshift'].values
        self.rv = self.z * constants.c / 1000
        self.mag = df['mag'].values
        self.snr00 = df['snr'].values
        if load_df: self.df = df
    
    def load_z(self, stage=None) -> None:
        load_path, num_samples = self.get_path_and_samples(stage)  
        df = pd.read_hdf(load_path)[:num_samples]          
        self.z = df['redshift'].values
        self.rv = self.z * constants.c / 1000
        
    def load_params(self, stage=None, load_df=False) -> None:
        load_path, num_samples = self.get_path_and_samples(stage)          
        df = pd.read_hdf(load_path)[:num_samples]          
        self.teff = df['T_eff'].values
        self.mh = df['M_H'].values
        self.am = df['a_M'].values
        self.cm = df['C_M'].values
        self.logg = df['log_g'].values
        if load_df: self.df = df
        # self.gd_labels = self.logg < 2.5
            
    def __getitem__(self, idx: int):
        return self.flux[idx], self.error[idx]

    def apply_mask(self):
        self.flux = self.fill_masked(self.flux, filler=self.mask_filler)        
        self.error = self.fill_masked(self.error, filler=self.mask_filler)        
        self.wave = self.wave[self.mask] if self.mask is not None else self.wave
        self.num_pixels = len(self.wave)
#endregion DATA-----------------------------------------------------------
#region DATATEST-----------------------------------------------------------
# if __name__ == '__main__':
#     d = BaseSpecMaskDataset.from_config({'data': {'file_path': './tests/spec/test_dataset.h5', 'num_samples': 10000}, 'mask': {'mask_ratio': 0.9}})
#     wave, flux, error = d.load_spec()
#     s = d.process_sigma(error, ratio=d.mask_ratio)
#     print(wave.shape, flux.shape, error.shape, s.shape)
#endregion DATASET-----------------------------------------------------------
#region DM-----------------------------------------------------------
class BaseDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 256, num_workers: int = 24, debug: bool = False, dataset_cls: Optional[Type['BaseDataset']] = None, config: Dict[str, Any] = {}):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.debug = debug
        self.dataset_cls = dataset_cls
        self.config = config
        self.reload = True

    @classmethod
    def from_config(cls, dataset_cls=BaseDataset, config: Dict[str, Any]={}):
        train_config = config.get('train', {})
        return cls(
            batch_size=train_config.get('batch_size', 256),
            num_workers=train_config.get('num_workers', 24),
            debug=train_config.get('debug', False),
            dataset_cls=dataset_cls,
            config=config,
        )

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train = self.dataset_cls.from_config(self.config)
            self.train.load_data()
            self.val = self.setup_test_dataset(stage='val')
            self.val.load_data(stage='val')
        elif stage == 'test':
            if self.reload:
                self.test = self.setup_test_dataset(stage)
                self.test.load_data(stage=stage)
            
    def setup_test_dataset(self):
        test_data, test_params = self.train.get_testdata()
        self.test_dict.update(test_params)
        return TensorDataset(*test_data)

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            shuffle=not self.debug
            # shuffle=False,
        )
    
    def val_dataloader(self):
        return DataLoader(self.val,
                          batch_size=len(self.val),
                          num_workers=self.num_workers,
                          shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          persistent_workers=True,
                          shuffle=False) 
#endregion DM-----------------------------------------------------------

#region MODEL-----------------------------------------------------------
class BaseModel(nn.Module, ABC):
    def __init__(self, model_name='model', loss_name='train'):
        super(BaseModel, self).__init__()
        self._model_name = model_name
        self._loss_name = loss_name
        print(f'Creating {self._model_name} model with {self._loss_name} loss')
        
    @property
    def name(self):
        return self._model_name
    @property
    def loss_name(self):
        return self._loss_name
    
    @abstractmethod
    def forward(self, x, labels=None):
        pass
    @abstractmethod
    def compute_loss(self, *args, **kwargs):
        pass
    @abstractmethod
    def log_outputs(self, outputs, log_fn = print, stage=''):
        log_fn(f'{self.loss_name}_loss', outputs['loss'])
#endregion MODEL-----------------------------------------------------------

class BaseLightningModule(L.LightningModule):
    def __init__(self, model: BaseModel = None, data_module = None, config={}):
        super().__init__()
        self.model = model
        self.data_module = data_module   # processed data
        self.loss_name = model.loss_name 
        self.callbacks = []
        self.config = config
        self.sweep = False

    def configure_optimizers(self):
        opt_config = {**self.config.get('opt', {}), 'loss_name': self.loss_name}
        return OptModule.from_config(opt_config)(self.model)
    
    def on_fit_start(self):
        
        if self.logger and hasattr(self.logger, 'experiment') and hasattr(self.logger.experiment, 'log'):
            num_params = sum(p.numel() for p in self.model.parameters())
            self.logger.experiment.log({"num_params(M)": round(num_params / 1e6, 3)})
            
    def on_train_epoch_start(self):
        """Log the current learning rate at the start of each epoch."""
        optimizer = self.trainer.optimizers[0]  # Assuming one optimizer
        lr = optimizer.param_groups[0]['lr']
        
        if self.logger and hasattr(self.logger, 'experiment') and hasattr(self.logger.experiment, 'log'):
            self.logger.experiment.log({"lr": lr})

   
class BaseTrainer(L.Trainer):
    def __init__(self, config, logger=False, num_gpus=None, sweep=False):
        if sweep:
            num_gpus = 1
            enable_checkpointing = False
            enable_progress_bar = False
            enable_model_summary = False
        else:
            enable_checkpointing = True
            enable_progress_bar = True
            enable_model_summary = True
            num_gpus = num_gpus or config.get('gpus', None) or torch.cuda.device_count()
        epoch = config.get('ep', 10)
        super().__init__(
            max_epochs=epoch,
            devices=num_gpus,
            accelerator='gpu',
            strategy='ddp' if num_gpus > 1 else 'auto',
            logger = logger,
            precision = '32',
            gradient_clip_val=config.get('grad_clip', 0.5),
            # stochastic_weight_avg=True,
            fast_dev_run=config.get('debug', False),
            enable_checkpointing=enable_checkpointing,
            enable_progress_bar=enable_progress_bar,
            enable_model_summary=enable_model_summary
        )
        
        
class PlotCallback(L.pytorch.callbacks.Callback):
    def __init__(self, interval=2, plotter=None, folder_name='plot', root_dir='./'):
        super().__init__()
        self.interval = interval
        self.p = plotter
        self.folder_name = folder_name
        self.dir = os.path.join(root_dir, self.folder_name)
        os.makedirs(self.dir, exist_ok=True)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        return cls(**{k: v for k, v in config.items() if k in cls.__init__.__code__.co_varnames})

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.plot_interval == 0:
            img_dict = self.plot_epoch_results(trainer, pl_module)
            for key, fig in img_dict.items():
                self.save_plot(fig, key, logger=None)

    def on_fit_end(self, trainer, pl_module):
        if trainer.logger and hasattr(trainer.logger, 'log_image'):
            self.logger = trainer.logger

        # self.create_gif()

    def plot_epoch_results(self, trainer, pl_module):
        pl_module.eval()
        with torch.no_grad():
            # Assuming your test data is stored in pl_module.test_dataset
            test_dataloader = pl_module.test_dataloader()
            batch = next(iter(test_dataloader))
            output_dict = pl_module.test_step(batch)
            self.p.plot(output_dict)
            img_dict = pl_module.model.plot(batch, output_dict,)
        return img_dict

    def save_plot(self, fig, name, logger = None):
        if logger is None:
            fig.savefig(os.path.join(self.dir, f'{name}.png'))
        else:
            logger.log_image(name, fig)

class OptModule():
    def __init__(self, lr, monitor_name='train_loss', opt_type='adam', weight_decay=0.0, lr_scheduler_name=None, **kwargs) -> None:
        self.lr = float(lr)
        self.monitor_name=monitor_name
        self.opt_type = opt_type
        self.weight_decay = weight_decay
        self.lr_scheduler_name = lr_scheduler_name
        self.kwargs = kwargs
        self.opt_fns = {
            'adam': torch.optim.Adam,
            'sgd': torch.optim.SGD,
            'rmsprop': torch.optim.RMSprop,
            'adadelta': torch.optim.Adadelta,
            'adagrad': torch.optim.Adagrad,
            'adamw': torch.optim.AdamW,
            'adamax': torch.optim.Adamax,
            'asgd': torch.optim.ASGD,
            'lbfgs': torch.optim.LBFGS,
            'rprop': torch.optim.Rprop,
            'sparseadam': torch.optim.SparseAdam,
        }
        self.lr_schedulers = {
            'cosine': torch.optim.lr_scheduler.CosineAnnealingLR,
            'onecycle': torch.optim.lr_scheduler.OneCycleLR,
            'plateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
        }

    @classmethod
    def from_config(cls, config):
        """
        Create an OptModule instance from a configuration dictionary.
        
        Args:
            config (dict): A dictionary containing the configuration parameters.
                           Expected keys: 'lr', 'type' (optional), 'lr_scheduler' (optional)
        
        Returns:
            OptModule: An instance of the OptModule class
        """
        lr = config.get('lr', 1e-3)
        opt_type = config.get('type', 'adam')
        weight_decay = config.get('weight_decay', 0)
        loss_name = config.get('loss_name', 'train')
        monitor_name = loss_name + '_loss'
        if 'lr_sch' in config:
            lr_scheduler_name = config['lr_sch']
            kwargs = {k: v for k, v in config.items() if k not in ['lr', 'type', 'lr_sch','weight_decay', 'loss_name']}
            return cls(lr=lr, monitor_name=monitor_name, opt_type=opt_type, weight_decay=weight_decay, lr_scheduler_name=lr_scheduler_name, **kwargs)
        return cls(lr=lr, monitor_name=monitor_name, opt_type=opt_type, weight_decay=weight_decay)

    def __call__(self, model):
        optimizer = self.opt_fns[self.opt_type](model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        if self.lr_scheduler_name is None:
            return optimizer
        if self.lr_scheduler_name not in self.lr_schedulers:
            raise ValueError(f"Unknown scheduler: {self.lr_scheduler_name}")
        scheduler = self.lr_schedulers[self.lr_scheduler_name](optimizer, **self.kwargs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, 'monitor': f'{self.monitor_name}'}

