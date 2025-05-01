import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from src.plotter import SpecPlotter
from src.utils import calculate_rms, calculate_snr, create_new_voigt_line, add_new_line, air_to_vac, get_equivalent_width
from torch.utils.data import Dataset, TensorDataset, DataLoader
from src.basemodule import BaseModel, BaseLightningModule, BaseTrainer, BaseSpecDataset, BaseDataModule, SingleSpectrumNoiseDataset

SAVE_DIR='/datascope/subaru/user/swei20/wandb'
SAVE_PATH = '/home/swei20/SirenSpec/checkpoints'
MASK_PATH = '/datascope/subaru/user/swei20/model/bosz50000_mask.npy'

#region --DATA-----------------------------------------------------------
class SpecTrainDataset(BaseSpecDataset):
    def load_data(self, stage=None) -> None:
        super().load_data(stage=stage)
        if self.mask_ratio is not None:
            if self.mask_ratio < 1:
                self.mask = np.load(MASK_PATH)
                self.apply_mask()
    
class SpecTestDataset(BaseSpecDataset):
    @classmethod
    def from_dataset(cls, dataset, stage='test'):
        keys = ['file_path', 'val_path', 'test_path', 'num_samples', 'num_test_samples', 'root_dir', 'mask_ratio', 'mask_filler', 'mask', 'lvrg_num', 'lvrg_mask', 'noise_level', 'noise_max']
        c = cls(**{k: getattr(dataset, k) for k in keys}) 
        if stage == 'val': c.num_test_samples = min(c.num_test_samples, 1000) 
        return c
    def load_data(self, stage=None) -> None:
        super().load_data(stage=stage)
        if self.mask is None and self.mask_ratio is not None:
            if self.mask_ratio < 1:
                self.mask = np.load(MASK_PATH)
            # self.mask = self.create_quantile_mask(self.error, ratio=self.mask_ratio)
        if self.mask is not None: 
            self.mask_plot = {'wave': self.wave, 'error':self.error[0], 'mask': self.mask}
            self.apply_mask()
            self.mask_plot.update({'masked_error': self.error[0]})       
        self.set_noise()    
        
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.noisy[idx], self.flux[idx], self.error[idx]
    
    def set_noise(self, seed=42):
        torch.manual_seed(seed)
        self.noise = torch.randn_like(self.flux) * self.error * self.noise_level
        self.noisy = self.flux + self.noise
        self.flux_rms = torch.norm(self.flux, dim=-1)
        self.snr0 = torch.div(self.flux_rms , torch.norm(self.noise, dim=-1))
        
    def get_single_spectrum_noise_testset(self, sample_idx=0, repeat=1000, seed=42):
        flux_0, error_0  = self.flux[sample_idx], self.error[sample_idx]
        test_dataset = SingleSpectrumNoiseDataset(flux_0, error_0, noise_level=self.noise_level,repeat=repeat, seed=seed)
        return test_dataset
    
#endregion --DATA-----------------------------------------------------------
#region --DATAMODULE-----------------------------------------------------------
class SpecDataModule(BaseDataModule):
    @classmethod
    def from_config(cls, config):
        return super().from_config(dataset_cls=SpecTrainDataset, config=config)
    def setup_test_dataset(self, stage):
        if hasattr(self, 'train'):
            return SpecTestDataset.from_dataset(self.train, stage) 
        return SpecTestDataset.from_config(self.config)
#endregion --DATAMODULE-----------------------------------------------------------

#region MODEL-----------------------------------------------------------
class AE(BaseModel):
    init_params = ['input_channels', 'output_channels', 'num_layers', 'embed_dim', 'kernel_size', 'loss_config']
    def __init__(self, input_channels=1, output_channels=2, num_layers=6, embed_dim=12, kernel_size=3, loss_config={'name': 'L1'}):
        pass
    
class Conv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding='same', use_bn=False, use_act=True, dilation=1):
        super(Conv1D, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.act = nn.LeakyReLU(negative_slope=0.1) if use_act else nn.Identity()
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else nn.Identity() 
        self.ofs = (kernel_size - 1) * dilation // 2  # adjust ofs for dilation
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='leaky_relu' if use_act else 'linear') #https://pytorch.org/docs/stable/nn.init.html
        nn.init.zeros_(self.conv.bias)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class CausalConv1D(Conv1D):
    def forward(self, x): # Blindspot by padding only on the left
        x = self.conv(F.pad(x, (self.ofs, 0)))[:, :, :-self.ofs]
        return self.act(self.bn(x))

class BlindspotModel1D(BaseModel):
    init_params = ['input_channels', 'output_channels', 'num_layers', 'embed_dim', 'kernel_size', 'input_sigma', 'use_bn', 'blindspot', 'dilation', 'loss_config']
    def __init__(self, input_channels=1, output_channels=2, num_layers=6, embed_dim=12, kernel_size=3, input_sigma=True, use_bn=False, blindspot=True, dilation=1, loss_config={'name': 'T1'}):
        # if output_channels is None: self.output_channels = input_channels + input_channels * (input_channels + 1) // 2
        self.num_layers = min(num_layers, 11) # can only pool log2(input_pixels)
        self.input_channels = input_channels  if not input_sigma else input_channels + 1
        self.input_sigma = input_sigma
        self.use_bn = use_bn
        self.T2 = loss_config.get('T2', 0)

        self.blindspot = blindspot
        assert 'name' in loss_config
        loss_name = loss_config.get('name', 'T1')
        name = f'b{int(blindspot)}_l{num_layers}_e{embed_dim}_k{kernel_size}_s{int(input_sigma)}_bn{int(use_bn)}_d{dilation}'
        super(BlindspotModel1D, self).__init__(model_name=name, loss_name=loss_name)
        conv_class = CausalConv1D if blindspot else Conv1D
        first_layer_class = conv_class
        unet_bn = use_bn
        self.encoders = nn.ModuleList([first_layer_class(self.input_channels, embed_dim, kernel_size=kernel_size, dilation=1, use_bn=unet_bn)] + 
                                      [conv_class(embed_dim, embed_dim, kernel_size=kernel_size, dilation=dilation, use_bn=unet_bn) for _ in range(num_layers)])
        self.decoders_a = nn.ModuleList([conv_class(embed_dim * 2, embed_dim, kernel_size=kernel_size, dilation=1, use_bn=unet_bn) for _ in range(num_layers - 2)] + 
                                        [conv_class(embed_dim + self.input_channels, embed_dim, kernel_size=kernel_size, dilation=1, use_bn=unet_bn)])
        self.decoders_b = nn.ModuleList([conv_class(embed_dim, embed_dim, kernel_size=kernel_size, dilation=1, use_bn=unet_bn) for _ in range(num_layers - 1)])
        
        nin_input_dim = embed_dim * 2 if blindspot else embed_dim  # 2 for left and right directions.
        nin_output_dim = output_channels if blindspot else 1
        self.nin_layers = nn.Sequential(  # kernel size= 1 for channel mixing
            Conv1D(nin_input_dim, nin_input_dim * 2, kernel_size=1, use_bn=use_bn),
            Conv1D(nin_input_dim * 2, embed_dim * 2, kernel_size=1, use_bn=use_bn),
            # Conv1D(embed_dim * 2, nin_output_dim, kernel_size=1, use_act=False),
            Conv1D(embed_dim * 2, nin_output_dim, kernel_size=1, use_bn=use_bn),
        )
        if blindspot: 
            self.nin_layers.append(nn.Softplus())

    def pool(self, x):
        if self.blindspot: x = F.pad(x[:, :, :-1], (1, 0))        
        return F.max_pool1d(x, kernel_size=2, stride=2, padding = 0)

    def unet(self, x):
        pools = [(x.size(-1), x)]
        x = self.encoders[0](x)
        for encoder_layer in self.encoders[1:-1]:
            x = encoder_layer(x)
            x = self.pool(x)
            pools.append((x.size(-1), x))
        pools.pop()
        x = self.encoders[-1](x)

        for (decoder_a, decoder_b) in zip(self.decoders_a, self.decoders_b):
            skip_size, skip_x = pools.pop()
            x = F.interpolate(x, size = skip_size , mode='nearest')
            concat = torch.cat([x, skip_x], dim=1)
            x = decoder_b(decoder_a(concat)) 
        return x

    def forward(self, x):
        batch_size = x.size(0) 
        if x.size(1) != self.input_channels: x = x.unsqueeze(1)   #size(x) = B，C，L
        if self.blindspot:         
            x = torch.cat([x, x.flip(-1)], dim=0)                 #size(x) = 2B，C，L
        x = self.unet(x)
        if self.blindspot:
            x = F.pad(x[:, :, :-1], (1, 0, 0, 0))
            x = torch.cat([x[:batch_size], x[batch_size:].flip(-1)], dim=1)
        return self.nin_layers(x)
        
    
    @classmethod
    def from_config(cls, model_config={}, loss_config={}):
        model_params = {k: model_config[k] for k in cls.init_params if k in model_config}
        return cls(**model_params, loss_config=loss_config)

    @classmethod
    def from_config_to_noise_estimator(cls, model_config={}):
        model_params = {k: model_config[k] for k in cls.init_params if k in model_config}
        model_params['output_channels'] = 1
        model_params['blindspot'] = False  
        return cls(**model_params, loss_config={})
   
    def weighted_l1_loss(self, y_pred, y_true, sigma_noise):
        l1_loss_all = F.l1_loss(y_pred, y_true, reduction='none')
        return l1_loss_all, torch.div(l1_loss_all, sigma_noise).mean()

    def compute_loss(self, noisy_signal, outputs, sigma_noise=None, labels=None, loss_only=False):
        if self.blindspot:
            return self.compute_blindspot_loss(noisy_signal, outputs, sigma_noise, labels, loss_only)
        else:
            return self.compute_unet_loss(noisy_signal, outputs, sigma_noise, labels, loss_only)
        
    def compute_unet_loss(self, noisy_signal, outputs, sigma_noise=None, labels=None, loss_only=False):
        clean_signal, error = labels
        denoised = outputs.squeeze(1)
        if sigma_noise is None: sigma_noise = error.clamp(min=1e-6)
        _, loss = self.weighted_l1_loss(denoised, clean_signal, sigma_noise) 
        if loss_only: return loss      
        
        with torch.no_grad():
            log_dict = {
                'snr0': calculate_rms(noisy_signal, clean_signal).mean(),
                'snr': calculate_rms(denoised, clean_signal).mean(),
                'mu_x': denoised.mean(),
                # 'sigma_x': sigma_x.mean(),
            }
        return  {'outputs': outputs, 'loss': loss, 'log_dict': log_dict, 'denoised': denoised}
        
    
    def compute_blindspot_loss(self, noisy_signal, outputs, sigma_noise=None, labels=None, loss_only=False):
        clean_signal, error = labels
        mu_x, sigma_x = outputs.split(1, dim=1)
        mu_x, sigma_x = mu_x.squeeze(1), sigma_x.squeeze(1)

        if sigma_noise is None: 
            sigma_noise = error
        else:
            sigma_loss = F.l1_loss(error, sigma_noise, reduction='mean')
            
        var_x, var_noise = sigma_x ** 2, sigma_noise ** 2
        var_y = var_x + var_noise
        if 'E2' in self.loss_name:
            loss = nn.GaussianNLLLoss(reduction='mean')(noisy_signal, mu_x, var_y)
        elif 'E1' in self.loss_name:
            loss = self.laplace_loss(noisy_signal, mu_x, var=var_y)
        # supervised loss
        denoised = torch.div((var_x * noisy_signal + var_noise * mu_x), var_y) 
        if 'T1' in self.loss_name:
            loss = self.laplace_loss(denoised, clean_signal, var=var_y)
        if self.T2 > 0:
            base_loss = loss
            T2_loss = self.T2 * nn.GaussianNLLLoss(reduction='mean')(denoised, clean_signal, var_y)
            loss = loss + T2_loss

        if loss_only: return loss      
          
        with torch.no_grad():
            l1_loss_all, wl1_loss = self.weighted_l1_loss(denoised, clean_signal, sigma_noise)
            log_dict = {
                'L1_loss': l1_loss_all.mean(),
                'WL1_loss': wl1_loss,
                'snr0': calculate_rms(noisy_signal, clean_signal).mean(),
                'snr': calculate_rms(denoised, clean_signal).mean(),
                'mu_x': mu_x.mean(),
                'sigma_x': sigma_x.mean(),
            }
            if self.T2 > 0:
                log_dict.update({'T2_loss': T2_loss, 'base_loss': base_loss})
        return  {'outputs': outputs, 'loss': loss, 'log_dict': log_dict, 'denoised': denoised}

    def compute_nd_loss(self, noisy_signal, outputs, sigma_noise, labels=None, loss_only=False):
        pass

    def laplace_loss(self, input, target, var=None, sigma0=None):
        if var is not None: 
            sigma = torch.sqrt(var.clamp(min=1e-12) / 2.0)         # b = sqrt(var/2)
        elif sigma0 is not None:
            sigma = sigma0.clamp(min=1e-6) / np.sqrt(2.0)       # b = sigma / sqrt(2)
        
        log_det_term = torch.log(2 * sigma)                   # log(2b) 
        quad_term = torch.div((input - target).abs(), sigma)  # |x - y| / b
        return (quad_term + log_det_term).mean()
    
    def log_outputs(self, outputs, log_fn=print, stage=''):
        if isinstance(outputs, dict):
            log_fn({f'{self.loss_name}_loss': outputs['loss']}, sync_dist=True)
            log_fn({f'{stage}/{k}': v.item() for k, v in outputs['log_dict'].items()}, sync_dist=True)
        else:
            log_fn({f'{self.loss_name}_loss': outputs})
#endregion
#region --TRAINER-----------------------------------------------------------
import wandb
import numpy as np
       
class SpecLModule(BaseLightningModule):
    def __init__(self, model=None, config={}, data_module=None):
        model = model or self.get_model(config)
        data_module = data_module or SpecDataModule.from_config(config)
        self.input_sigma = config.get('model', {}).get('input_sigma', False)
        self.use_denoised = False     
        self.is_last_epoch = False
        self.noise_level = config['noise'].get('noise_level', 0.0)
        super().__init__(model=model, data_module=data_module, config=config)
        
        self.denoised = {}
        self.valid_dict = {'snr0': [], 'snr': [], 'ca_snr0': [],  'ca_snr': [], 'denoised': []}
        self.run_name = ""
        self.artifact = None   
         
    def get_model(self, config):
        model_config = config.get('model', {})
        model_name = model_config.get('name', 'blindspot')
        if model_name == 'blindspot':
            self.fix_sigma = model_config.get('fix_sigma', True)
            if not self.fix_sigma:
                self.noise_estimator = BlindspotModel1D.from_config_to_noise_estimator(model_config=config.get('model', {}))
            return BlindspotModel1D.from_config(model_config=model_config, loss_config =config.get('loss', {}))
        # elif model_name == 'ae':
        #     return AE.from_config(model_config)
    
    def forward(self, noisy, flux, error_nl, loss_only=False):
        outputs = self.model(noisy)
        return self.model.compute_loss(noisy, outputs, labels=(flux, error_nl), loss_only=loss_only)
     
    def training_step(self, batch, batch_idx):
        flux, error = batch
        noisy = flux + torch.randn_like(flux) * error * self.noise_level
        loss = self(noisy, flux, error * self.noise_level, loss_only=True)
        self.log(f'{self.loss_name}_loss', loss, sync_dist=True)
        return loss
    
    def on_validation_epoch_start(self):
        self.is_last_epoch = self.current_epoch == self.trainer.max_epochs - 1
        
    def validation_step(self, batch, batch_idx):
        noisy, flux, error = batch
        output_dict = self(noisy, flux, error * self.noise_level, loss_only=False)
        self.model.log_outputs(output_dict, log_fn=self.log_dict, stage='val')
        if batch_idx == 0:
            self.valid_dict['snr'].append(output_dict['log_dict']['snr'].detach().cpu().numpy())
            if self.is_last_epoch:
                self.valid_dict['snr0'].append(output_dict['log_dict']['snr0'].detach().cpu().numpy())
                self.valid_dict['denoised'].append(output_dict['denoised'].detach().cpu())
                self.valid_output_dict = output_dict
        return output_dict['loss']
   
    def on_validation_epoch_end(self):
        if self.is_last_epoch:
            # if self.logger and hasattr(self.logger, 'experiment'):
            #     self.logger.experiment.log({f"valid/snr_hist": wandb.Histogram([self.valid_dict['snr'], self.valid_dict['snr0']], num_bins=100)})
            self.data_module.val.denoised = torch.cat(self.valid_dict['denoised'], dim=0)
            self.vplotter = SpecPlotter(self.data_module.val)
            val_id = 3
            val_fig = self.vplotter.plot_idx(val_id)
            self.log_fig({f'train/spec{val_id}': wandb.Image(val_fig)})

    def on_test_start(self):
        self.test_dict = {'snr0': [], 'snr': [], 'ca_snr0': [],  'ca_snr': [], 'denoised': []}
        self.wave = self.data_module.test.wave
        self.ca_rng = [8475, 8680]
        self.ca_mask = (self.wave >= self.ca_rng[0]) & (self.wave <= self.ca_rng[1])

    def test_step(self, batch, batch_idx):
        noisy, flux, error = batch
        output_dict = self(noisy, flux, error * self.noise_level, loss_only=False)
        self.test_dict['snr0'].append(output_dict['log_dict']['snr0'].detach().cpu().numpy())
        self.test_dict['snr'].append(output_dict['log_dict']['snr'].detach().cpu().numpy())
        self.test_dict['ca_snr'].append(calculate_snr(output_dict['denoised'][..., self.ca_mask]).mean().detach().cpu().numpy())
        self.test_dict['denoised'].append(output_dict['denoised'].detach().cpu())
        self.model.log_outputs(output_dict, log_fn=self.log_dict, stage='test')
        if batch_idx == 0:
            self.test_output_dict= output_dict

    def on_test_epoch_end(self):
        self.data_module.test.denoised = torch.cat(self.test_dict['denoised'], dim=0)
        self.data_module.test.snr = calculate_rms(flux=self.data_module.test.flux, noisy=self.data_module.test.denoised)
        snr_fig, axs = plt.subplots(1, 2, figsize=(10, 3))
        data = [d for k, d in self.test_dict.items() if k in ['snr', 'snr0', 'ca_snr']]
        labels = [f'{k} {np.mean(v):.0f}' for k, v in self.test_dict.items() if k in ['snr', 'snr0', 'ca_snr']]
        color = ['k', 'b', 'darkorange']
        _ = axs[0].hist(data, bins=50, label = labels, color=color, density=True)
        axs[0].set_xlabel('SNR Distribution')
        
        # if self.valid_dict['snr']:
        #     final_valid_snr = self.valid_dict['snr'][-1]
        #     axs[0].axvline(final_valid_snr, color='darkred', linestyle='--', label=f'val snr{final_valid_snr:.0f}')
        #     axs[1].plot(self.valid_dict['snr'],'o-', color='darkred', label='Valid snr vs epoch', )
        #     axs[1].set_xlabel('epoch')
        #     axs[1].set_xlim(0, len(self.valid_dict['snr']))
        # for ax in axs: ax.legend(loc = 'upper left')
        # self.log_fig({'test/snr_hist': wandb.Image(snr_fig)})

        self.tplotter = SpecPlotter(self.data_module.test)
        test_id = 815
        test_fig = self.tplotter.plot_idx(test_id)
        self.log_fig({f'test/spec{test_id}': wandb.Image(test_fig)})
        self.data_module.test.load_snr(stage='test')
        eq_ca2_snr2_fig = self.tplotter.plot_equivalent_width()
        self.log_fig({f'test/eq_ca2_snr2': wandb.Image(eq_ca2_snr2_fig)})

        try:
            eq_violin_fig=self.tplotter.plot_equivalent_width_violin()
            self.log_fig({f'test/eq_violin_fig': wandb.Image(eq_violin_fig)})

            ew_std_fig = self.tplotter.plot_ew_std_comparison()
            self.log_fig({f'test/ew_std_fig': wandb.Image(ew_std_fig)})

            snr_improve_fig, snr_improv_log_fig = self.tplotter.plot_snr_improve()
            self.log_fig({f'test/snr_improve_fig': wandb.Image(snr_improve_fig)})
            self.log_fig({f'test/snr_improv_log_fig': wandb.Image(snr_improv_log_fig)})
        except:
            print('merp')
            
    def log_fig(self, fig_dict):
        if self.logger and hasattr(self.logger.experiment, 'log'):
            self.logger.experiment.log(fig_dict)
        else:
            pass
    def calculate_snr(self, flux):
        return calculate_snr(flux)
    def calculate_rms(self, noisy=None, flux=None, residual=None):
        return calculate_rms(noisy=noisy, flux=flux, residual=residual)
    
    def quick_test(self, noisy, flux, error, noise_level=None, outputs=False):
        if noise_level is None: noise_level = self.noise_level
        ds = TensorDataset(noisy, flux, error)
        denoised_all = []
        outputs_all = []
        with torch.no_grad():
            for batch in DataLoader(ds, batch_size=128):
                noisy, flux, error = batch
                noisy = noisy.to(self.device)
                flux = flux.to(self.device)
                error = error.to(self.device)

                output_dict = self(noisy, flux, error * noise_level, loss_only=False)
                denoised_all.append(output_dict['denoised'].detach().cpu())
                if outputs: 
                    outputs_all.append(output_dict['outputs'].detach().cpu())
                    return torch.cat(denoised_all, dim=0), torch.cat(outputs_all, dim=0)
        return torch.cat(denoised_all, dim=0)
    
class BlindspotLModule(SpecLModule):      
    def forward(self, noisy, flux, error, loss_only=False):
        inputs = torch.cat([noisy.unsqueeze(1), error.unsqueeze(1)], dim=1) if self.input_sigma else noisy
        outputs = self.model(inputs)      
        sigma_noise = None if self.fix_sigma else self.noise_estimator(noisy)
        return self.model.compute_loss(noisy, outputs, sigma_noise=sigma_noise, labels=(flux, error), loss_only=loss_only)
    
#endregion --TRAINER-----------------------------------------------------------

import lightning as L

class SpecTrainer():
    def __init__(self, config, logger, num_gpus=None, sweep=False) -> None:
        if sweep: num_gpus = 1
        snr_patience = 100 if sweep else 500
        self.trainer = BaseTrainer(config=config.get('train', {}), logger=logger, num_gpus=num_gpus, sweep=sweep)  #
        if not sweep: 
            checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(dirpath= SAVE_PATH, filename='{epoch}-{snr_valid:.0f}', save_top_k=1, monitor='val/snr', mode='max')
            self.trainer.callbacks.append(checkpoint_callback)
            
        earlystopping_callback = L.pytorch.callbacks.EarlyStopping(monitor='val/snr', patience=snr_patience, mode='max', divergence_threshold=1,)
        self.trainer.callbacks.append(earlystopping_callback)
        self.test_trainer = L.Trainer(devices=1, accelerator='gpu', logger=logger,  enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)

class Experiment:
    def __init__(self, config, use_wandb=False, num_gpus=None, sweep=False, ckpt_path=None):
        self.lightning_module = BlindspotLModule(config=config)
        self.lightning_module.sweep = sweep
        if use_wandb:
            if sweep:
                logger = L.pytorch.loggers.WandbLogger(config=config, name=self.lightning_module.model.name, log_model=False, save_dir=SAVE_DIR) 
            else:
                logger = L.pytorch.loggers.WandbLogger(project = config['project'], config=config, name=self.lightning_module.model.name, log_model=True, save_dir=SAVE_DIR)
        else:
            logger = None
        self.t = SpecTrainer(config = config, logger = logger, num_gpus=num_gpus, sweep=sweep)
        self.ckpt_path = ckpt_path
    
    def run(self):
        self.t.trainer.fit(self.lightning_module, datamodule=self.lightning_module.data_module, ckpt_path=self.ckpt_path)
        self.t.test_trainer.test(self.lightning_module, datamodule=self.lightning_module.data_module)
    
if __name__ == '__main__':
    config = {
        'loss': {'name': 'E1'},
        'data': {'file_path': './tests/spec/test_dataset.h5', 'num_samples': 10,},
        'mask': {'mask_ratio': 0.9, },
        'noise': {'noise_level': 2.0, },
        'train': {'ep': 2},
        'model': {'input_sigma': True, 'blindspot': True, 'num_layers': 3, 'embed_dim': 3, 'kernel_size': 3}
    }
    exp = Experiment(config, use_wandb=False, num_gpus=1)
    exp.run()