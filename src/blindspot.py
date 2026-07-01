"""BlindSpotDenoiser — 1D blindspot UNet for stellar spectra.

BEFORE MODIFYING THIS FILE, READ `AGENT_BRIEFING.md` in the repo root.

Key constraint: channel 2 of the input MUST be invariant to per-pixel
flux permutation. In the simulated setting, error[i] = sqrt(flux[i] + sky)
is a near-lossless flux proxy; passing per-pixel error as input defeats
the blindspot guarantee. Use per-spectrum broadcast scalars
(sigma_input_mode L0-L3) instead. See AGENT_BRIEFING.md.

Metric reporting: always report BOTH val/snr_mu_x (blindspot-pure) and
val/snr_denoised (posterior-fused). Monitor val/snr_mu_x for EarlyStopping
and Checkpoint. val/snr on its own is ambiguous and should not be reported.
"""
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
    
def _resolve_norm_type(norm_type, use_bn):
    """Back-compat helper: accept either explicit norm_type or legacy use_bn bool."""
    if norm_type is not None:
        return norm_type
    return 'bn' if use_bn else 'none'


class Conv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding='same',
                 norm_type=None, use_bn=False, use_act=True, dilation=1):
        super(Conv1D, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.act = nn.LeakyReLU(negative_slope=0.1) if use_act else nn.Identity()
        norm_type = _resolve_norm_type(norm_type, use_bn)
        if norm_type == 'bn':
            self.bn = nn.BatchNorm1d(out_channels)
        elif norm_type == 'ln':
            # GroupNorm with 1 group behaves like LayerNorm across channels for Conv1d
            self.bn = nn.GroupNorm(1, out_channels)
        else:
            self.bn = nn.Identity()
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
    # Physics-informed injection params (P1-P6) live in model_config.physics.*
    # and are passed through init_params so they serialize in state_dict metadata.
    init_params = ['input_channels', 'output_channels', 'num_layers', 'embed_dim', 'kernel_size',
                   'input_sigma', 'sigma_feature_channels', 'use_bn', 'norm_type', 'blindspot',
                   'dilation', 'loss_config',
                   'use_identity_residual', 'residual_gain', 'residual_gain_ramp_epochs',
                   'use_continuum_marginalize', 'continuum_basis', 'continuum_degree', 'lam_cont',
                   'use_doppler_deshift', 'doppler_method',
                   'use_template_bayes', 'compute_f_man',
                   'spectrum_length',
                   'output_mode', 'huber_delta']
    def __init__(self, input_channels=1, output_channels=2, num_layers=6, embed_dim=12, kernel_size=3,
                 input_sigma=True, sigma_feature_channels=1, use_bn=False, norm_type=None,
                 blindspot=True, dilation=1, loss_config={'name': 'T1'},
                 # === Physics-informed injections (P1-P6) — all default OFF ===
                 # P1 identity-preserving residual (Pro wMAE / Θ-SFD): mu_x = noisy + gain * delta
                 # residual_gain_ramp_epochs: if set, gain linearly 0→residual_gain over first N epochs,
                 # then held at residual_gain. Fixes v9/v10/v12 failure mode where hard gain=1.0 from
                 # scratch compresses gradient signal and NN fails to learn effective delta.
                 use_identity_residual=False, residual_gain=1.0, residual_gain_ramp_epochs=None,
                 # P3 continuum gauge marginalization (Pro §8.2): project residual onto high-freq
                 use_continuum_marginalize=False, continuum_basis='legendre',
                 continuum_degree=3, lam_cont=0.0,
                 # P2 Doppler log-lambda translation (Pro §8.1) — stub, needs RV label in datamodule
                 use_doppler_deshift=False, doppler_method='fft',
                 # P5 Template-Bayes Θ-D3' — stub, separate code path (scripts/run_template_bayes.py)
                 use_template_bayes=False,
                 # P4 F_man measured floor (Pro §1.3) — evaluator-only flag, no training effect
                 compute_f_man=False,
                 # Spectrum length for P3 basis precompute (PFS-MR = 4096)
                 spectrum_length=4096,
                 # === mu-only mode (drops posterior fusion, single-head output) ===
                 # 'mu_sigma' (default): Laine 2019 two-head + posterior fusion (legacy, bit-for-bit unchanged)
                 # 'mu'                : single-head mu_x; loss = Huber(mu_x, noisy_signal); denoised := mu_x
                 # See plan/2026-04-25 mu-only design (Bayes-optimal in high-SNR continuum-normalized regime).
                 output_mode='mu_sigma',
                 # Huber transition (quadratic→linear) anchored to noise scale σ_n; only used when output_mode='mu'.
                 huber_delta=0.05):
        # sigma_feature_channels: number of per-spectrum broadcast scalar channels appended to input
        # when input_sigma=True. Determined by sigma_input_mode: L0=1, L1=3, L2=1, L3=1.
        self.num_layers = min(num_layers, 11) # can only pool log2(input_pixels)
        self.input_channels = input_channels + (sigma_feature_channels if input_sigma else 0)
        self.input_sigma = input_sigma
        self.sigma_feature_channels = sigma_feature_channels
        self.use_bn = use_bn
        self.norm_type = _resolve_norm_type(norm_type, use_bn)
        self.T2 = loss_config.get('T2', 0)
        # Physics-informed regularizers (L0_v3+): 0.0 = disabled, preserves old behavior.
        self.lam_mu  = loss_config.get('lam_mu',  0.0)   # per-spectrum mean(mu_x) -> 0.5
        self.lam_sig = loss_config.get('lam_sig', 0.0)   # soft cap sigma_x <= sigma_noise
        self.lam_neg = loss_config.get('lam_neg', 0.0)   # flux non-negativity
        self.log_sigma_clamp_max = loss_config.get('log_sigma_clamp_max', 6.0)

        # === Physics-informed injection flags (P1-P6) ===
        self.use_identity_residual = bool(use_identity_residual)
        self.residual_gain = float(residual_gain)
        self.residual_gain_ramp_epochs = (
            int(residual_gain_ramp_epochs) if residual_gain_ramp_epochs else None
        )
        self._current_epoch = 0  # set by BlindspotLModule.on_train_epoch_start
        self.use_continuum_marginalize = bool(use_continuum_marginalize)
        self.continuum_basis_type = continuum_basis
        self.continuum_degree = int(continuum_degree)
        self.lam_cont = float(lam_cont)
        self.use_doppler_deshift = bool(use_doppler_deshift)
        self.doppler_method = doppler_method
        self.use_template_bayes = bool(use_template_bayes)
        self.compute_f_man_flag = bool(compute_f_man)
        self.spectrum_length = int(spectrum_length)

        if self.use_template_bayes:
            raise NotImplementedError(
                "P5 use_template_bayes=True: BlindspotModel1D is a point denoiser; "
                "Template-Bayes (Theta-D3') needs a separate forward-model chain. "
                "Use scripts/run_template_bayes.py (not yet implemented)."
            )
        if self.compute_f_man_flag:
            print("[P4] compute_f_man=True: evaluator-only flag. "
                  "Use scripts/compute_f_man.py for F_man = L^-1 tr[J H^-1 J^T].")

        # === mu-only mode (drops posterior fusion) ===
        if output_mode not in ('mu', 'mu_sigma'):
            raise ValueError(f"output_mode must be 'mu' or 'mu_sigma', got {output_mode!r}")
        self.output_mode = output_mode
        self.huber_delta = float(huber_delta)
        self.m1_target = loss_config.get('target', 'noisy')
        if self.m1_target not in ('noisy', 'clean'):
            raise ValueError(
                f"loss.target must be 'noisy' or 'clean' for M1, got {self.m1_target!r}"
            )
        if self.output_mode == 'mu':
            # mu-only is exclusive with all sigma_x-coupled regularizers and loss families.
            _bad = []
            if self.T2 > 0: _bad.append(f'T2={self.T2}')
            if self.lam_sig > 0: _bad.append(f'lam_sig={self.lam_sig}')
            if self.lam_cont > 0: _bad.append(f'lam_cont={self.lam_cont}')
            _ln = loss_config.get('name', '')
            if _ln in ('T1', 'T2', 'E1', 'E2'):
                _bad.append(f"loss.name={_ln!r}")
            if _bad:
                raise ValueError(
                    f"output_mode='mu' is exclusive with sigma_x-coupled config: {_bad}. "
                    f"Use loss.name='M1' and drop T2 / lam_sig / lam_cont. See plan/synchronous-meandering-pillow.md."
                )

        self.blindspot = blindspot
        assert 'name' in loss_config
        loss_name = loss_config.get('name', 'T1')
        name = f'b{int(blindspot)}_l{num_layers}_e{embed_dim}_k{kernel_size}_s{int(input_sigma)}_n{self.norm_type}_d{dilation}_o{output_mode}'
        super(BlindspotModel1D, self).__init__(model_name=name, loss_name=loss_name)
        conv_class = CausalConv1D if blindspot else Conv1D
        first_layer_class = conv_class
        unet_norm = self.norm_type
        self.encoders = nn.ModuleList([first_layer_class(self.input_channels, embed_dim, kernel_size=kernel_size, dilation=1, norm_type=unet_norm)] +
                                      [conv_class(embed_dim, embed_dim, kernel_size=kernel_size, dilation=dilation, norm_type=unet_norm) for _ in range(num_layers)])
        self.decoders_a = nn.ModuleList([conv_class(embed_dim * 2, embed_dim, kernel_size=kernel_size, dilation=1, norm_type=unet_norm) for _ in range(num_layers - 2)] +
                                        [conv_class(embed_dim + self.input_channels, embed_dim, kernel_size=kernel_size, dilation=1, norm_type=unet_norm)])
        self.decoders_b = nn.ModuleList([conv_class(embed_dim, embed_dim, kernel_size=kernel_size, dilation=1, norm_type=unet_norm) for _ in range(num_layers - 1)])

        nin_input_dim = embed_dim * 2 if blindspot else embed_dim  # 2 for left and right directions.
        # mu-only emits 1 channel (mu_x); mu_sigma emits output_channels (typically 2: mu + log_sigma).
        nin_output_dim = 1 if self.output_mode == 'mu' else (output_channels if blindspot else 1)
        self.nin_layers = nn.Sequential(  # kernel size= 1 for channel mixing
            Conv1D(nin_input_dim, nin_input_dim * 2, kernel_size=1, norm_type=unet_norm),
            Conv1D(nin_input_dim * 2, embed_dim * 2, kernel_size=1, norm_type=unet_norm),
            Conv1D(embed_dim * 2, nin_output_dim, kernel_size=1, norm_type=unet_norm),
        )
        # F6: Softplus removed. Output channel 2 is now log_sigma_x; sigma_x = exp(log_sigma_x)
        # is guaranteed positive and numerically stable without a nonlinearity on output.

        # === P3 Continuum gauge basis: LAZY init at first forward (Pro §8.2) ===
        # Real spectrum length at forward depends on mask_ratio (e.g. 4096 * 0.85 = 3481),
        # so we defer build to _ensure_continuum_basis(L, device) inside compute_blindspot_loss.
        # Stored as non-buffer tensor attributes (not in state_dict — recomputable).
        self._continuum_B = None
        self._continuum_B_pinv = None

    def _ensure_continuum_basis(self, L: int, device, dtype):
        """Build or rebuild Legendre basis (L, k) and pseudo-inverse (k, L) if shape/device mismatch."""
        need_rebuild = (
            self._continuum_B is None
            or self._continuum_B.shape[0] != L
            or self._continuum_B.device != device
            or self._continuum_B.dtype != dtype
        )
        if need_rebuild:
            B_c = self._build_continuum_basis(
                L, self.continuum_degree, basis_type=self.continuum_basis_type
            ).to(device=device, dtype=dtype)                             # (L, k)
            self._continuum_B = B_c
            self._continuum_B_pinv = torch.linalg.pinv(B_c)              # (k, L)

    @staticmethod
    def _build_continuum_basis(L: int, degree: int, basis_type: str = 'legendre') -> 'torch.Tensor':
        """Build (L, k) low-frequency basis for P3 continuum gauge (k = degree+1).

        Pro §8.2: continuum is a gauge nuisance; in loss, project residual onto
        span(B_c)^perp so continuum mismatch does not dominate T1.
        """
        x = torch.linspace(-1.0, 1.0, L, dtype=torch.float32)
        k = degree + 1
        if basis_type == 'legendre':
            B = torch.zeros(L, k, dtype=torch.float32)
            B[:, 0] = 1.0
            if k >= 2:
                B[:, 1] = x
            for n in range(1, degree):
                B[:, n + 1] = ((2 * n + 1) * x * B[:, n] - n * B[:, n - 1]) / (n + 1)
            return B
        elif basis_type == 'polynomial':
            return torch.stack([x ** i for i in range(k)], dim=-1)
        elif basis_type == 'fourier_lowfreq':
            components = [torch.ones(L, dtype=torch.float32)]
            for i in range(1, k):
                components.append(torch.cos(i * torch.pi * (x + 1) / 2.0))
            return torch.stack(components, dim=-1)
        else:
            raise ValueError(
                f"Unknown continuum_basis='{basis_type}'. "
                f"Supported: 'legendre', 'polynomial', 'fourier_lowfreq'."
            )

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
        # The audit experiments need a no-mask control. The Conv1D forward path
        # already supports blindspot=False; restrict it to mu-only so the
        # legacy mu_sigma branch cannot hit a one-channel output mismatch.
        if not self.blindspot and self.output_mode != 'mu':
            raise ValueError("blindspot=False is supported only with output_mode='mu'.")
        return self.compute_blindspot_loss(noisy_signal, outputs, sigma_noise, labels, loss_only)

    def compute_blindspot_loss(self, noisy_signal, outputs, sigma_noise=None, labels=None, loss_only=False):
        """Compute blindspot loss and metrics.

        F5 (log_sigma_x) + D6 (metric split):
        - Output channel 2 is now log_sigma_x; sigma_x = exp(log_sigma_x)
          (guaranteed positive, stable; no Softplus on output layer).
        - sigma_noise: caller (BlindspotLModule.forward) MUST pass a sigma that
          is invariant to per-pixel flux permutation. If None, fall back to
          per-spectrum mean (not raw error) — see AGENT_BRIEFING.md rule #4.
        - F4 (σ_post scale fix) is DEFERRED per plan D4: loss still uses
          var_y = var_x + var_noise as in paper's current eq:lnll. DO NOT change
          to var_post until paper numbers are out and Vika approves.
        - Always log BOTH snr_mu_x (blindspot-pure) and snr_denoised (fused).
          Monitor val/snr_mu_x for EarlyStop/Checkpoint.
        """
        clean_signal, error = labels

        # === mu-only branch (single-head; drops sigma_x / posterior fusion) ===
        if self.output_mode == 'mu':
            mu_x = outputs.squeeze(1)                       # (B, L)
            # P1 identity-preserving residual: NN learns small delta around noisy_signal.
            if self.use_identity_residual:
                if self.residual_gain_ramp_epochs:
                    ramp = min(1.0, (self._current_epoch + 1) / float(self.residual_gain_ramp_epochs))
                    effective_gain = self.residual_gain * ramp
                else:
                    effective_gain = self.residual_gain
                mu_x = noisy_signal + effective_gain * mu_x

            # Loss: Huber on noisy_signal (self-supervised N2V/N2N) or clean_signal
            # (supervised center-visible control when blindspot=False), selected by loss.target.
            if 'M1' not in self.loss_name:
                raise ValueError(
                    f"output_mode='mu' requires loss.name='M1', got {self.loss_name!r}. "
                    f"See plan/synchronous-meandering-pillow.md C2."
                )
            target_signal = clean_signal if self.m1_target == 'clean' else noisy_signal
            loss = F.smooth_l1_loss(mu_x, target_signal, beta=self.huber_delta)
            denoised = mu_x                                  # downstream eval uses 'denoised' as the user-facing output
            if self.lam_neg > 0:
                loss = loss + self.lam_neg * torch.relu(-mu_x).mean()

            if loss_only: return loss
            with torch.no_grad():
                l1_loss_all, wl1_loss = self.weighted_l1_loss(denoised, clean_signal, error.mean(dim=-1, keepdim=True).expand_as(error))
                snr_mu_x = calculate_rms(mu_x, clean_signal).mean()
                log_dict = {
                    'L1_loss': l1_loss_all.mean(),
                    'WL1_loss': wl1_loss,
                    'snr0': calculate_rms(noisy_signal, clean_signal).mean(),
                    'snr': snr_mu_x,
                    'snr_mu_x': snr_mu_x,
                    'snr_denoised': snr_mu_x,                # mu-only: denoised := mu_x; alias for back-compat with monitors/plotters
                    'mu_x': mu_x.mean(),
                }
            return {'loss': loss, 'denoised': denoised, 'mu_x': mu_x, 'outputs': outputs, 'log_dict': log_dict}

        # === legacy mu_sigma branch (Laine 2019 two-head + posterior fusion) ===
        mu_x, log_sigma_x = outputs.split(1, dim=1)
        mu_x, log_sigma_x = mu_x.squeeze(1), log_sigma_x.squeeze(1)

        # === P1 Identity-preserving residual (Pro wMAE / Θ-SFD) ===
        # Interpret NN output as delta; mu_x = noisy + gain * delta.
        # Low-noise regime: |delta| small → mu_x ≈ noisy → preserves high-SNR pixels.
        if self.use_identity_residual:
            if self.residual_gain_ramp_epochs:
                ramp = min(1.0, (self._current_epoch + 1) / float(self.residual_gain_ramp_epochs))
                effective_gain = self.residual_gain * ramp
            else:
                effective_gain = self.residual_gain
            mu_x = noisy_signal + effective_gain * mu_x

        # Clamp log_sigma_x to a sane range for numerical safety. exp(-12) ~ 6e-6,
        # exp(6) ~ 400 — wider than any realistic noise regime.
        log_sigma_x = log_sigma_x.clamp(min=-12.0, max=self.log_sigma_clamp_max)
        sigma_x = torch.exp(log_sigma_x)

        if sigma_noise is None:
            # NEW policy (AGENT_BRIEFING rule #4): do NOT fall back to raw per-pixel error.
            # Use per-spectrum mean as the safe, leakage-free fallback.
            sigma_noise = error.mean(dim=-1, keepdim=True).expand_as(error)

        var_x, var_noise = sigma_x ** 2, sigma_noise ** 2
        var_y = var_x + var_noise                                  # y|flux marginal variance — used by E1/E2 (supervises noisy_signal)
        var_post = var_x * var_noise / var_y.clamp(min=1e-12)      # conjugate-Gaussian posterior variance — used by T1/T2 (supervises denoised)
        if 'E2' in self.loss_name:
            loss = nn.GaussianNLLLoss(reduction='mean')(noisy_signal, mu_x, var_y)
        elif 'E1' in self.loss_name:
            loss = self.laplace_loss(noisy_signal, mu_x, var=var_y)
        # Posterior-mean fusion (Bayesian conjugate) — unchanged
        denoised = torch.div((var_x * noisy_signal + var_noise * mu_x), var_y)
        if 'T1' in self.loss_name:
            # === P3 Continuum gauge marginalization (Pro §8.2) ===
            # Project residual = denoised - clean onto span(B_c)^perp before T1 loss.
            # Low-freq (continuum gauge) residuals are nuisance and excluded from T1.
            # Optional soft L2 penalty via lam_cont keeps low-freq residual bounded.
            if self.use_continuum_marginalize:
                residual = denoised - clean_signal                                    # (B, L)
                self._ensure_continuum_basis(residual.shape[-1], residual.device, residual.dtype)
                # β = B_c^+ @ residual   →  (B, k)
                coefs = torch.einsum('kl,bl->bk', self._continuum_B_pinv, residual)
                residual_lowfreq = torch.einsum('lk,bk->bl', self._continuum_B, coefs)  # (B, L)
                residual_hf = residual - residual_lowfreq                             # gauge-projected
                loss = self.laplace_loss(residual_hf,
                                         torch.zeros_like(residual_hf),
                                         var=var_post)
                if self.lam_cont > 0:
                    loss = loss + self.lam_cont * (residual_lowfreq ** 2).mean()
            else:
                loss = self.laplace_loss(denoised, clean_signal, var=var_post)
        # Physics-informed regularizers (L0_v3+). Opt-in via loss_config.lam_*.
        if self.lam_mu > 0:
            per_spec_mean = mu_x.mean(dim=-1)                                    # (B,)
            loss = loss + self.lam_mu * ((per_spec_mean - 0.5) ** 2).mean()
        if self.lam_sig > 0:
            log_sig_noise = 0.5 * torch.log(var_noise.clamp(min=1e-12))          # log(sigma_noise)
            excess = torch.relu(log_sigma_x - log_sig_noise)                     # >0 only if sigma_x > sigma_noise
            loss = loss + self.lam_sig * (excess ** 2).mean()
        if self.lam_neg > 0:
            loss = loss + self.lam_neg * torch.relu(-mu_x).mean()
        if self.T2 > 0:
            base_loss = loss
            T2_loss = self.T2 * nn.GaussianNLLLoss(reduction='mean')(denoised, clean_signal, var_post)
            loss = loss + T2_loss

        if loss_only: return loss

        with torch.no_grad():
            l1_loss_all, wl1_loss = self.weighted_l1_loss(denoised, clean_signal, sigma_noise)
            snr_mu_x = calculate_rms(mu_x, clean_signal).mean()
            snr_denoised = calculate_rms(denoised, clean_signal).mean()
            log_dict = {
                'L1_loss': l1_loss_all.mean(),
                'WL1_loss': wl1_loss,
                'snr0': calculate_rms(noisy_signal, clean_signal).mean(),
                'snr_mu_x': snr_mu_x,          # blindspot-pure — PRIMARY metric
                'snr_denoised': snr_denoised,   # posterior-fused (reinjects noisy[i])
                'snr': snr_mu_x,                # back-compat alias → points to mu_x
                'mu_x': mu_x.mean(),
                'log_sigma_x': log_sigma_x.mean(),
                'sigma_x': sigma_x.mean(),
            }
            if self.T2 > 0:
                log_dict.update({'T2_loss': T2_loss, 'base_loss': base_loss})
        return  {'outputs': outputs, 'loss': loss, 'log_dict': log_dict, 'denoised': denoised, 'mu_x': mu_x}

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
            # Inject sigma_feature_channels derived from sigma_input_mode so
            # BlindspotModel1D sets input_channels correctly before conv build.
            sigma_input_mode = model_config.get('sigma_input_mode', 'L1')
            mc = dict(model_config)
            mc['sigma_feature_channels'] = resolve_sigma_feature_channels(sigma_input_mode)
            if not self.fix_sigma:
                self.noise_estimator = BlindspotModel1D.from_config_to_noise_estimator(model_config=mc)
            return BlindspotModel1D.from_config(model_config=mc, loss_config=config.get('loss', {}))
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
        test_id = min(815, len(self.data_module.test) - 1)
        try:
            test_fig = self.tplotter.plot_idx(test_id)
            self.log_fig({f'test/spec{test_id}': wandb.Image(test_fig)})
        except Exception as e:
            print(f'[plot] plot_idx failed (non-fatal): {e}')
        try:
            self.data_module.test.load_snr(stage='test')
            eq_ca2_snr2_fig = self.tplotter.plot_equivalent_width()
            self.log_fig({f'test/eq_ca2_snr2': wandb.Image(eq_ca2_snr2_fig)})
        except Exception as e:
            print(f'[plot] plot_equivalent_width failed (non-fatal): {e}')

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
    
_SIGMA_FEATURE_CHANNELS = {'none': 0, 'L0': 1, 'L1': 3, 'L2': 1, 'L3': 1, 'L4': 1, 'L5': 1}


def resolve_sigma_feature_channels(sigma_input_mode):
    """Number of channels produced by _process_error_input for a given mode.

    Used by config loaders and Experiment.__init__ to set input_channels
    correctly before building the model.
    """
    if sigma_input_mode not in _SIGMA_FEATURE_CHANNELS:
        raise ValueError(f"Unknown sigma_input_mode: {sigma_input_mode}")
    return _SIGMA_FEATURE_CHANNELS[sigma_input_mode]


class BlindspotLModule(SpecLModule):
    """Blindspot Lightning module — simulated-data-safe input pipeline.

    sigma_input_mode (see AGENT_BRIEFING.md, plan F1):
      none : no second channel; loss uses per-spectrum mean(error) fallback
      L0   : 1 scalar — sqrt(mean(error^2)) broadcast to L positions
      L1   : 3 scalars — [rms, median, p90] broadcast
      L2   : coarse wavelength bins — mean(error) per bin, bin-broadcast
             (sigma_input_bins bins, each ≥ 256 wide; enforced at init)
      L3   : wide lowpass — avg_pool1d(error, k=sigma_input_lowpass)
             (kernel must be ≥ 501)
      L4   : per-pixel train-set average profile — σ̄_i = mean_over_train_spec(error_i).
             A fixed (L,) tensor loaded from sigma_profile_path; broadcast to (B, 1, L).
             Does NOT depend on current spectrum's error, so T1 permute is trivially safe.
             T2 (error→ones) will NOT change channel — this is by design (L4 ignores input error channel).
      L5   : per-pixel avg × per-spectrum scalar — σ̄_i × sqrt(mean(error_of_this_spec^2)).
             Global shape from train set, per-spec amplitude from this spec's rms(error).
             Permutation-invariant scalar (rms) so T1 safe.

    OBSOLETE modes (deleted, plan F10): 'raw', 'smoothed', 'global', 'blindspot'
    (sigma_mode enum). All raw-per-pixel-error inputs are banned because
    error[i] ≈ sqrt(flux[i]+sky) is a deterministic flux proxy in simulated data.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_cfg = self.config.get('model', {})
        self.sigma_input_mode = model_cfg.get('sigma_input_mode', 'L1')
        self.sigma_input_bins = model_cfg.get('sigma_input_bins', 16)
        self.sigma_input_lowpass = model_cfg.get('sigma_input_lowpass', 501)
        self.sigma_profile_path = model_cfg.get('sigma_profile_path', None)
        if self.sigma_input_mode not in _SIGMA_FEATURE_CHANNELS:
            raise ValueError(
                f"sigma_input_mode='{self.sigma_input_mode}' invalid. "
                f"Must be one of {list(_SIGMA_FEATURE_CHANNELS.keys())}. "
                f"Obsolete 'raw'/'smoothed'/'global'/'blindspot' sigma_modes are removed — "
                f"see AGENT_BRIEFING.md and plan F10."
            )
        # L4/L5 need a precomputed per-pixel profile tensor.
        if self.sigma_input_mode in ('L4', 'L5'):
            if self.sigma_profile_path is None:
                raise ValueError(
                    f"sigma_input_mode='{self.sigma_input_mode}' requires model.sigma_profile_path. "
                    f"Generate via scripts/compute_sigma_profile.py."
                )
            profile = torch.load(self.sigma_profile_path, map_location='cpu')
            if not isinstance(profile, torch.Tensor):
                profile = torch.as_tensor(profile)
            profile = profile.float().flatten()
            self.register_buffer('sigma_profile', profile)

    def on_train_epoch_start(self):
        # Sync current_epoch into the underlying model so P1 gain ramp can read it.
        if hasattr(self, 'model') and hasattr(self.model, '_current_epoch'):
            self.model._current_epoch = int(self.current_epoch)
        super().on_train_epoch_start() if hasattr(super(), 'on_train_epoch_start') else None

    def _process_error_input(self, error):
        """Return a (B, C_sigma, L) tensor where per-pixel flux is NOT recoverable.

        See AGENT_BRIEFING.md for why per-pixel error is banned.
        """
        mode = self.sigma_input_mode
        if mode == 'none' or not self.input_sigma:
            return None
        B, L = error.shape
        if mode == 'L0':
            s = torch.sqrt((error ** 2).mean(dim=-1, keepdim=True))  # (B, 1)
            return s.unsqueeze(-1).expand(B, 1, L)  # (B, 1, L)
        if mode == 'L1':
            rms = torch.sqrt((error ** 2).mean(dim=-1, keepdim=True))      # (B, 1)
            med = error.median(dim=-1, keepdim=True).values                # (B, 1)
            p90 = error.quantile(0.9, dim=-1, keepdim=True)                # (B, 1)
            stacked = torch.cat([rms, med, p90], dim=-1)                   # (B, 3)
            return stacked.unsqueeze(-1).expand(B, 3, L)                   # (B, 3, L)
        if mode == 'L2':
            bins = self.sigma_input_bins
            bin_size = L // bins
            assert bin_size >= 256, (
                f"L2 bin_size={bin_size} < 256 — too small to prevent per-pixel leak. "
                f"Reduce sigma_input_bins (L={L})."
            )
            padded_len = bins * bin_size
            err_trunc = error[:, :padded_len].reshape(B, bins, bin_size)
            bin_means = err_trunc.mean(dim=-1, keepdim=True)              # (B, bins, 1)
            broadcast = bin_means.expand(B, bins, bin_size).reshape(B, padded_len)
            if padded_len < L:
                pad = broadcast[:, -1:].expand(B, L - padded_len)
                broadcast = torch.cat([broadcast, pad], dim=-1)
            return broadcast.unsqueeze(1)                                 # (B, 1, L)
        if mode == 'L3':
            k = self.sigma_input_lowpass
            assert k >= 501, "L3 lowpass kernel must be >= 501 to avoid per-pixel leak"
            # avg_pool1d produces output length = L - k + 1 with stride=1, no padding.
            # Use padding=k//2 (symmetric reflect-like) to preserve length L.
            padded = F.pad(error.unsqueeze(1), (k // 2, k // 2), mode='replicate')
            out = F.avg_pool1d(padded, kernel_size=k, stride=1)
            if out.shape[-1] > L:
                out = out[..., :L]
            elif out.shape[-1] < L:
                out = F.pad(out, (0, L - out.shape[-1]), mode='replicate')
            return out                                                    # (B, 1, L)
        if mode == 'L4':
            # Per-pixel train-set avg profile, fixed, broadcast to batch.
            # Does not depend on current spectrum — T1 permute safe by construction.
            profile = self.sigma_profile.to(error.device)                 # (L,)
            assert profile.shape[0] == L, (
                f"sigma_profile length {profile.shape[0]} != L {L}. "
                f"Regenerate profile for this dataset."
            )
            return profile.view(1, 1, L).expand(B, 1, L)                  # (B, 1, L)
        if mode == 'L5':
            profile = self.sigma_profile.to(error.device)                 # (L,)
            assert profile.shape[0] == L, (
                f"sigma_profile length {profile.shape[0]} != L {L}. "
                f"Regenerate profile for this dataset."
            )
            # per-spec scalar rms(error) — permutation-invariant so T1 safe
            scalar = torch.sqrt((error ** 2).mean(dim=-1, keepdim=True))  # (B, 1)
            # normalize profile to unit mean so scalar controls absolute amplitude
            profile_n = profile / profile.mean().clamp_min(1e-12)         # (L,)
            out = profile_n.view(1, 1, L) * scalar.unsqueeze(-1)          # (B, 1, L)
            return out.expand(B, 1, L)
        raise ValueError(f"Unknown sigma_input_mode: {mode}")

    def _get_loss_sigma(self, error, processed_sigma):
        """Return a (B, L) tensor used as σ_0 in the loss / posterior fusion.

        MUST NOT carry per-pixel flux info — same constraint as input channel
        (AGENT_BRIEFING.md rule #4).
        """
        mode = self.sigma_input_mode
        if mode == 'none':
            # No per-spectrum info available as a channel; still safe to use
            # per-spectrum mean (a single scalar broadcast, invariant to permutation).
            return error.mean(dim=-1, keepdim=True).expand_as(error)
        if processed_sigma is None:
            return error.mean(dim=-1, keepdim=True).expand_as(error)
        if processed_sigma.shape[1] == 1:
            return processed_sigma.squeeze(1)
        # L1 has 3 channels — first channel is rms (per-spectrum scalar broadcast).
        return processed_sigma[:, 0, :]

    def forward(self, noisy, flux, error, loss_only=False):
        # === P6 σ-not-as-input ===
        # When config sets model.input_sigma=False, the σ channel(s) are NOT concatenated
        # to the NN input; σ still enters the loss via sigma_noise (_get_loss_sigma).
        # This respects Pro §6 single-spine: σ in loss, never input.
        processed_sigma = self._process_error_input(error)  # (B, C_sigma, L) or None
        if processed_sigma is not None and self.input_sigma:
            inputs = torch.cat([noisy.unsqueeze(1), processed_sigma], dim=1)
        else:
            inputs = noisy.unsqueeze(1)

        # === P2 Doppler log-lambda de-shift (Pro §8.1) — STUB ===
        # Requires SpecTrainDataset to load redshift label from HDF5
        # (spectrumdataset/params/table col 0) and plumb it through forward.
        # Not yet wired; explicit NotImplementedError prevents silent misuse.
        if getattr(self.model, 'use_doppler_deshift', False):
            raise NotImplementedError(
                "P2 use_doppler_deshift=True requires RV-aware datamodule "
                "(SpecTrainDataset must yield redshift label). "
                "Stubbed for future implementation — see plan/2026-04-21-physics-injection.md."
            )

        outputs = self.model(inputs)
        # CRITICAL: sigma_noise used in loss / posterior must also be the PROCESSED sigma,
        # not raw per-pixel error. See AGENT_BRIEFING.md rule #4.
        sigma_noise = self._get_loss_sigma(error, processed_sigma)
        return self.model.compute_loss(
            noisy, outputs, sigma_noise=sigma_noise, labels=(flux, error), loss_only=loss_only
        )
    
#endregion --TRAINER-----------------------------------------------------------

import lightning as L

class SpecTrainer():
    def __init__(self, config, logger, num_gpus=None, sweep=False) -> None:
        if sweep: num_gpus = 1
        train_cfg = config.get('train', {})
        snr_patience = train_cfg.get('early_stop_patience', 100 if sweep else 500)
        self.trainer = BaseTrainer(config=train_cfg, logger=logger, num_gpus=num_gpus, sweep=sweep)  #
        # F7: monitor val/snr_mu_x (blindspot-pure), not val/snr (fused — can short-circuit via noisy[i]).
        if not sweep:
            ckpt_dir = train_cfg.get('ckpt_dir', SAVE_PATH)   # override per-run to avoid overwrites across parallel experiments
            checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(dirpath=ckpt_dir, filename='{epoch}-{val/snr_mu_x:.0f}', save_top_k=1, monitor='val/snr_mu_x', mode='max')
            self.trainer.callbacks.append(checkpoint_callback)

        # divergence_threshold: set to 1 historically (resume-from-peak runs had snr>1 from ep0).
        # From-scratch runs need this disabled (ep 0-2 snr<1 is normal). Config can override:
        #   train:
        #     divergence_threshold: null  # disable (new from-scratch runs)
        #     divergence_threshold: 1     # legacy behavior
        divergence_threshold = train_cfg.get('divergence_threshold', 1)
        earlystopping_callback = L.pytorch.callbacks.EarlyStopping(monitor='val/snr_mu_x', patience=snr_patience, mode='max', divergence_threshold=divergence_threshold,)
        self.trainer.callbacks.append(earlystopping_callback)
        self.test_trainer = L.Trainer(devices=1, accelerator='gpu', logger=logger,  enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)

class Experiment:
    def __init__(self, config, use_wandb=False, num_gpus=None, sweep=False, ckpt_path=None, init_from=None):
        seed = config.get('train', {}).get('seed', None)
        if seed is not None:
            L.seed_everything(int(seed), workers=True)
            print(f'[seed] Lightning seed_everything({seed}), workers=True')
        self.lightning_module = BlindspotLModule(config=config)
        if init_from:
            import torch as _torch
            sd = _torch.load(init_from, map_location='cpu', weights_only=False)['state_dict']
            missing, unexpected = self.lightning_module.load_state_dict(sd, strict=False)
            print(f'[init_from] loaded weights from {init_from}')
            print(f'[init_from] missing={len(missing)} unexpected={len(unexpected)}')
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
