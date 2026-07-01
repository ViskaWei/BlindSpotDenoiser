import yaml
import torch
import numpy as np
import pandas as pd

lick = {
    'TiO_4': [7643.25, 7717.25, 7527.0, 7577.75, 7735.5, 7782.75],
    # 'NaI_V12': [8180.0, 8200.0, 8164.0, 8173.0, 8233.0, 8244.0],
    # 'NaI_F13': [8180.0, 8200.0, 8137.0, 8147.0, 8233.0, 8244.0],
    # 'NaI_LB13': [8180.0, 8200.0, 8143.0, 8153.0, 8233.0, 8244.0],
    'Ca1_LB13': [8484.0, 8513.0, 8474.0, 8484.0, 8563.0, 8577.0],
    'Ca2_LB13': [8522.0, 8562.0, 8474.0, 8484.0, 8563.0, 8577.0],
    'Ca3_LB13': [8642.0, 8682.0, 8619.0, 8642.0, 8700.0, 8725.0]}

lick_vac = {'TiO_4': np.array([7645.45, 7719.47, 7529.16, 7579.93, 7737.72, 7784.99]),
 'Ca1_LB13': np.array([8486.44, 8515.45, 8476.44, 8486.44, 8565.46, 8579.47]),
 'Ca2_LB13': np.array([8524.45, 8564.46, 8476.44, 8486.44, 8565.46, 8579.47]),
 'Ca3_LB13': np.array([8644.49, 8684.5 , 8621.48, 8644.49, 8702.5 , 8727.51])}

# import astropy.units as u
# from specutils import Spectrum1D, SpectralRegion
# from specutils.manipulation import extract_region
# def get_equivalent_width_i(spectrum, feature_start, feature_end, blue_start, blue_end, red_start, red_end, factor=1):
#     blue_reg = extract_region(spectrum, SpectralRegion(blue_start*u.AA, blue_end*u.AA))
#     feat_reg = extract_region(spectrum, SpectralRegion(feature_start*u.AA, feature_end*u.AA))
#     red_reg  = extract_region(spectrum, SpectralRegion(red_start*u.AA,  red_end*u.AA))

#     blue_mean_flux = np.mean(blue_reg.flux)
#     red_mean_flux  = np.mean(red_reg.flux)
#     blue_midwave = np.mean(blue_reg.spectral_axis)
#     red_midwave  = np.mean(red_reg.spectral_axis)

#     def continuum_model(wave_array):
#         m = (red_mean_flux - blue_mean_flux) / (red_midwave - blue_midwave)
#         b = blue_mean_flux - m * blue_midwave
#         return m * wave_array + b

#     feat_wave = feat_reg.spectral_axis
#     feat_flux = feat_reg.flux
#     cont_flux = continuum_model(feat_wave)

#     # 4) Integrate: EW = ∫ [1 - (F_line / F_cont)] dλ
#     integrand = 1 - (feat_flux / cont_flux)
#     ew_value = np.trapz(integrand.value, x=feat_wave.value)  # dimensionless × Å
#     ew_quantity = ew_value * feat_wave.unit
#     return ew_quantity

def get_equivalent_width_i(wave, flux, feature_start, feature_end, blue_start, blue_end, red_start, red_end):
     # 1) Select the regions (blue, feature, red) via boolean masks
    blue_mask = (wave >= blue_start) & (wave <= blue_end)
    feat_mask = (wave >= feature_start) & (wave <= feature_end)
    red_mask  = (wave >= red_start) & (wave <= red_end)
    # 2) Compute mean flux and mean wave in blue and red regions
    blue_mean_flux = torch.mean(flux[blue_mask])
    red_mean_flux  = torch.mean(flux[red_mask])
    blue_midwave   = torch.mean(wave[blue_mask])
    red_midwave    = torch.mean(wave[red_mask])
    # 3) Build a simple linear continuum model
    def continuum_model(wave):
        m = (red_mean_flux - blue_mean_flux) / (red_midwave - blue_midwave)  #slope
        b = blue_mean_flux - m * blue_midwave #intercept
        return m * wave + b
    # 4) Extract the feature region data
    feat_wave = wave[feat_mask]
    feat_flux = flux[feat_mask]
    # 5) Evaluate the continuum in the feature region
    cont_flux = continuum_model(feat_wave)
    # 6) Integrate: EW = ∫ [1 - (F_line / F_cont)] dλ
    integrand = 1.0 - (feat_flux / cont_flux)
    # implement trapezoid integration manually.
    ew_value = torch.trapz(integrand, feat_wave)
    return ew_value.numpy()

from tqdm import tqdm
def get_equivalent_width(wave, all_spectra, all_zs):
    results_list = []
    for i, spec in tqdm(enumerate(all_spectra)):
        row_data = {"SpecID": i}
        for idx_name, rngs in lick_vac.items():
            rngs_z = rngs * (1 + all_zs[i])
            ew = get_equivalent_width_i(wave, spec, *rngs_z)
            row_data[idx_name] = ew  # store just the numeric part; or store ew.to_string()
        results_list.append(row_data)
    df_results = pd.DataFrame(results_list)
    return df_results


from scipy.special import voigt_profile
def z0(X, z):
    return X / (1 + z)

def create_new_voigt_line(input_wave, u=8700, hw=10, sigma=1.0, gamma=1.0):
    new_wave = input_wave[(input_wave > u-hw) & (input_wave < u+hw)]
    new_line = voigt_profile(new_wave - u, sigma, gamma)
    # new_line = voigt_profile(new_wave - np.mean(new_wave), sigma, gamma)
    return new_wave, new_line

def add_new_line(old_wave, old_flux, new_wave, new_line, sign=1):
    idx = np.where(np.isin(old_wave, new_wave))
    line_flux = old_flux[idx] + new_line * sign
    new_flux = old_flux.copy()
    new_flux[idx] = line_flux
    return line_flux, new_flux


class SVDDenoiser:
    """
    SVD-based denoiser that learns the basis from clean spectra and can be applied to noisy spectra.
    """
    def __init__(self, n_components):
        """
        Initialize the denoiser.
        
        Args:
            n_components (int): Number of SVD components to keep
        """
        self.n_components = n_components
        self.V = None  # Right singular vectors (learned basis)
        self.mean_spectrum = None
        self.explained_variance_ratio_ = None
    
    def fit(self, clean_spectra):
        """
        Learn the SVD basis from clean spectra.
        
        Args:
            clean_spectra (torch.Tensor): Clean spectra matrix of shape (n_samples, n_features)
        """
        # Center the data
        self.mean_spectrum = clean_spectra.mean(dim=0)
        centered_spectra = clean_spectra - self.mean_spectrum
        
        # Perform SVD on clean data
        U, S, V = torch.svd(centered_spectra)
        
        # Store the truncated basis
        self.V = V[:, :self.n_components]
        
        # Calculate explained variance ratio
        total_var = (S ** 2).sum()
        self.explained_variance_ratio_ = (S ** 2)[:self.n_components] / total_var
        
        return self
    
    def denoise(self, noisy_spectra):
        """
        Denoise spectra using the learned basis.
        
        Args:
            noisy_spectra (torch.Tensor): Noisy spectra matrix of shape (n_samples, n_features)
            
        Returns:
            torch.Tensor: Denoised spectra
        """
        if self.V is None:
            raise RuntimeError("Must fit the denoiser before using it to denoise spectra")
        
        # Center the noisy data using the mean from training
        centered_noisy = noisy_spectra - self.mean_spectrum
        
        # Project onto the learned basis
        coefficients = centered_noisy @ self.V
        
        # Reconstruct using the learned basis
        denoised_centered = coefficients @ self.V.T
        
        # Add back the mean
        denoised_spectra = denoised_centered + self.mean_spectrum
        
        return denoised_spectra
    
    def fit_transform(self, clean_spectra):
        """
        Learn the basis and denoise the clean spectra in one step.
        
        Args:
            clean_spectra (torch.Tensor): Clean spectra matrix
            
        Returns:
            torch.Tensor: Denoised clean spectra
        """
        self.fit(clean_spectra)
        return self.denoise(clean_spectra)
    
@staticmethod
def calculate_snr(flux):
    signal = torch.median(flux, dim=-1).values
    diff = 2 * flux[..., 1:-1] - flux[..., :-2] - flux[..., 2:]
    coeff = 1.482602 / torch.sqrt(torch.tensor(6.0)) 
    noise = coeff * torch.median(diff.abs(), dim=-1).values
    return torch.div(signal, noise)

@staticmethod
def calculate_rms(noisy=None, flux=None, residual=None):
    if residual is None:
        residual = noisy - flux
    return torch.norm(flux, dim=-1)  / torch.norm(residual, dim=-1) 


def calculate_snr_np(value, mask, shift=1, binning=1):
    signal = np.median(value[mask])
    df = np.abs(2 * value[mask][shift:-shift] - value[mask][:-2 * shift] - value[mask][2 * shift:])
    noise = 1.482602 / np.sqrt(6.0) * np.median(df)
    snr = signal / noise * np.sqrt(binning)
    return snr

@staticmethod
def masked_var_fraction(noise, mask):
    total_variance = torch.var(noise).item()
    masked_variance = torch.var(noise[mask]).item()
    # unmasked_variance = torch.var(noise[~mask]).item()
    
    # Calculate fraction of total variance from masked regions
    masked_fraction = (masked_variance * mask.sum()) / (total_variance * len(noise)) * 100
    return masked_fraction


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def save_model(model, filepath):
    """
    Save a PyTorch model to a file.
    
    Args:
    model (torch.nn.Module): The PyTorch model to save
    filepath (str): The path where the model will be saved
    """
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")
    

def load_model(model, filepath):
    """
    Load a PyTorch model from a file.
    
    Args:
    model (torch.nn.Module): An instance of the model architecture
    filepath (str): The path where the model is saved
    
    Returns:
    torch.nn.Module: The loaded model
    """
    model.load_state_dict(torch.load(filepath))
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {filepath}")
    return model

# Example usage:
# Assuming 'model' is your trained model and 'RTokenEmbedding' is your model class

# Saving the model
# save_model(model, 'rtokenembedding_model.pth')

# Loading the model (for future use)
# loaded_model = load_model(RTokenEmbedding(params...), 'rtokenembedding_model.pth')

# def get_fc_input(self, image_size, padding=0):
#     conv_output = math.floor((image_size - self.kernel_size + 2 * padding) / self.stride + 1)
#     return conv_output * conv_output


# import os
# import re
# import imageio



# def create_gif_from_pngs(input_folder, output_gif, duration=2, last_frame_duration=10):
#     """
#     Create a GIF from PNG images in the input folder.
    
#     :param input_folder: Path to the folder containing PNG images
#     :param output_gif: Path to save the output GIF
#     :param duration: Duration of each frame in the GIF (in seconds)
#     """
#     # Get all PNG files in the input folder
#     png_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')],
#                        key=lambda x: int(re.search(r'ep(\d+)', x).group(1)))
    
#     images = []
#     durations = []
#     for png_file in png_files:
#         file_path = os.path.join(input_folder, png_file)
#         images.append(imageio.imread(file_path))
#         durations.append(duration)
    
#     # Set the last frame's duration to 10 seconds
#     durations[-1] = last_frame_duration
    
#     # Save the images as a GIF
#     imageio.mimsave(output_gif, images, duration=duration, loop=0)
#     print(f"GIF created and saved as {output_gif}")

# import numpy as np
# from PIL import Image


# def create_standardized_gif(input_folder, output_gif, duration=2, target_size=None, last_frame_duration=10):
#     # Get all PNG files in the folder
#     png_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')])
#         # png_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')],
#         #                key=lambda x: int(re.search(r'ep(\d+)', x).group(1)))
#     if not png_files:
#         print(f"No PNG files found in {input_folder}")
#         return

#     images = []
    
#     # Determine target size if not provided
#     if target_size is None:
#         with Image.open(os.path.join(input_folder, png_files[0])) as img:
#             target_size = img.size

#     print(f"Standardizing images to size: {target_size}")
#     durations = []

#     for png_file in png_files:
#         with Image.open(os.path.join(input_folder, png_file)) as img:
#             # Resize and convert to RGB
#             img_resized = img.resize(target_size, Image.LANCZOS).convert('RGB')
#             images.append(np.array(img_resized))
#             durations.append(duration)
#     durations[-1] = last_frame_duration

#     # Save as GIF
#     imageio.mimsave(output_gif, images, duration=duration, loop=0)
#     print(f"GIF created: {output_gif}")

# def create_gif(input_folder, output_gif, duration=2, last_frame_duration=10):
#     try:
#         create_gif_from_pngs(input_folder, output_gif, duration, last_frame_duration)
#     except Exception as e:
#         create_standardized_gif(input_folder, output_gif, duration, last_frame_duration)
        

from torch.optim.lr_scheduler import StepLR

def train(model, train_loader, criterion, optimizer, num_epochs=1, device='cpu', lr_decay_step=100, lr_decay_gamma=0.1):
    model.train()
    _ = get_num_params(model)

    model.to(device)
    scheduler = StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)

    for epoch in range(num_epochs):
        total_loss = 0.0
        for x in train_loader:
            x = [xi.to(device) for xi in x]
            # [print(xi.shape) for xi in x]
            optimizer.zero_grad()            
            output = model(*x[:-1])
            loss = criterion(output, x[-1])

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6e}")
        scheduler.step()

    return output, x


def get_num_params(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params/1e6:.1f}M")
    return total_params

@staticmethod
def air_to_vac(wave):
    """
    Implements the air to vacuum wavelength conversion described in eqn 65 of
    Griesen 2006
    """
    wlum = wave * 1e5
    return (1 + 1e-6 * (287.6155 + 1.62887 / wlum**2 + 0.01360 / wlum**4)) * wave


@staticmethod
def vac_to_air(wave):
    fact = 1.0 + 2.735182e-4 + 131.4182 / wave**2 + 2.76249e8 / wave**4
    fact = fact * (wave >= 2000) + 1.0 * (wave < 2000)
    return wave/fact

@staticmethod
def air_to_vac_deriv(wave):
    """
    Eqn 66
    """
    wlum = wave * 1e5
    return (1+1e-6*(287.6155 - 1.62877/wlum**2 - 0.04080/wlum**4))
