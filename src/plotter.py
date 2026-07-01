#region --PLOT-----------------------------------------------------------
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde, skew
from src.utils import calculate_rms, calculate_snr, get_equivalent_width
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
import pandas as pd
import torch

class SpecPlotter():
    def __init__(self, dataset):
        self.d = dataset
        self.wave = dataset.wave
        self.ca_rng = [8475, 8680]
        self.ca_mask = self.get_mask_from_wave_range(self.ca_rng)
        self.dpi = 100
        self.df_equivalent_width = None

    def get_mask_from_wave_range(self, wave_range):
        return (self.wave >= wave_range[0]) & (self.wave <= wave_range[1])

    @staticmethod
    def _snr_db(flux, estimate):
        residual = flux - estimate
        var_flux = flux.var(unbiased=True).clamp(min=1e-30)
        var_res = residual.var(unbiased=True).clamp(min=1e-30)
        return float((10.0 * torch.log10(var_flux / var_res)).item())

    def _format_parameter_text(self, i):
        parts = []

        def _value(name):
            if not hasattr(self.d, name):
                return None
            values = getattr(self.d, name)
            try:
                value = values[i]
            except Exception:
                return None
            if value is None or pd.isna(value):
                return None
            return float(value)

        teff = _value('teff')
        mh = _value('mh')
        logg = _value('logg')
        mag = _value('mag')
        snr_meta = _value('snr00')
        if snr_meta is None:
            snr_meta = _value('snr_meta')

        if teff is not None:
            parts.append(rf"$T_{{\rm eff}}={teff:.0f}$")
        if mh is not None:
            parts.append(rf"$M_{{\rm H}}={mh:.1f}$")
        if logg is not None:
            parts.append(rf"$\log g={logg:.1f}$")
        if mag is not None:
            parts.append(rf"$\mathrm{{mag}}={mag:.1f}$")
        if snr_meta is not None:
            parts.append(rf"$\mathrm{{S/N}}_{{\rm meta}}={snr_meta:.1f}$")
        return "   ".join(parts) if parts else None

    def plot_idx(self, i, lw=0.8):
        # Create figure with 1 row and 5 columns (left, middle-top, middle-bottom, right-top, right-bottom)
        fig = plt.figure(figsize=(16, 6), dpi=self.dpi)
        gs = fig.add_gridspec(
            2,
            3,
            height_ratios=[1, 1],
            width_ratios=[1.08, 1, 1],
            wspace=0.006,
            hspace=0.012,
        )
        
        x, y, d = self.d.flux[i], self.d.noisy[i], self.d.denoised[i]
        
        snr0 = self.d.snr0[i] if hasattr(self.d, 'snr0') else calculate_rms(flux=x, noisy=y)
        snr = calculate_rms(flux=x, noisy=d)
        ca_snr0 = calculate_rms(flux=x[self.ca_mask], noisy=y[self.ca_mask])
        ca_snr = calculate_rms(flux=x[self.ca_mask], noisy=d[self.ca_mask])
        snr0_db = self._snr_db(x, y)
        snr_db = self._snr_db(x, d)
        ca_snr_db = self._snr_db(x[self.ca_mask], d[self.ca_mask])
        
        # Left plot (stays single)
        ax0 = fig.add_subplot(gs[:, 0])  # Takes all rows in first column
        ax0.plot(self.wave, y, c='k', alpha=1., lw=lw-0.2)
        ax0.plot(self.wave, x, c='r', lw=lw)
        ax0.axvspan(*self.ca_rng, color='orange', alpha=0.3)
        ax0.set(xlim=(self.wave[0], self.wave[-1]), 
            xlabel='Wavelength (A)', 
            ylabel='Normalized Flux', 
            ylim=(-0., 1.))
        
        # Middle plots (top and bottom)
        ax1_top = fig.add_subplot(gs[0, 1])  # Top middle
        ax1_bottom = fig.add_subplot(gs[1, 1])  # Bottom middle
        offset = 0.2
        
        # Top middle plot (x)
        ax1_top.plot(self.wave, x, c='r', lw=lw)
        ax1_top.plot(self.wave, d-offset, c='b', lw=lw)
        ax1_top.axvspan(*self.ca_rng, color='orange', alpha=0.3)
        ax1_top.set(xlim=(self.wave[0], self.wave[-1]), 
                # ylim=(-0.12, 0.55),
                xticklabels=[])
        
        # Bottom middle plot (x - d)
        ax1_bottom.plot(self.wave, x - d, c='teal', lw=lw)
        ax1_bottom.axvspan(*self.ca_rng, color='orange', alpha=0.3)
        ax1_bottom.set(xlim=(self.wave[0], self.wave[-1]), 
                    xlabel='Wavelength (A)',)
                    # ylim=(-0.12, 0.55))
        
        # Right plots (top and bottom)
        ax2_top = fig.add_subplot(gs[0, 2], sharey=ax1_top)  # Top right
        ax2_bottom = fig.add_subplot(gs[1, 2], sharey=ax1_bottom)  # Bottom right
        
        # Top right plot (x)
        ax2_top.plot(self.wave, x, c='r', lw=lw)
        ax2_top.plot(self.wave, d-offset, c='b', lw=lw)
        ax2_top.axvspan(*self.ca_rng, color='orange', alpha=0.3)
        ax2_top.set(xlim=(self.ca_rng[0]-10, self.ca_rng[-1]+10),
                # ylim=(-0.12, 0.55),
                xticklabels=[])
        ax2_top.yaxis.tick_right()
        ax2_top.yaxis.set_label_position("right")
        ax2_top.set_ylabel('Normalized Flux')
        ax2_top.tick_params(axis='y', left=False, labelleft=False, right=True, labelright=True)

        
        # Bottom right plot (x - d)
        ax2_bottom.plot(self.wave, x - d, c='teal', lw=lw)
        ax2_bottom.axvspan(*self.ca_rng, color='orange', alpha=0.3)
        ax2_bottom.set(xlim=(self.ca_rng[0]-10, self.ca_rng[-1]+10),
                    xlabel='Wavelength (A)',)
                    # ylim=(-0.12, 0.55))
        ax2_bottom.yaxis.tick_right()
        ax2_bottom.yaxis.set_label_position("right")
        ax2_bottom.set_ylabel(r'$x-\mu$')
        ax2_bottom.tick_params(axis='y', left=False, labelleft=False, right=True, labelright=True)
        
        header_y = 0.9985
        legend_lw = 8.0
        handles = [
            Line2D([], [], color='k', lw=legend_lw, label=r'$y$'),
            Line2D([], [], color='r', lw=legend_lw, label=r'$x$'),
            Line2D([], [], color='b', lw=legend_lw, label=fr'$\mu - {offset:.1f}$'),
            Line2D([], [], color='teal', lw=legend_lw, label=r'$x-\mu$'),
            Patch(facecolor='orange', edgecolor='none', alpha=0.3, label='Ca II Region'),
        ]
        param_text = self._format_parameter_text(i)
        if param_text is not None:
            fig.text(0.008, header_y, param_text, ha='left', va='top', fontsize=12.2)
            fig.legend(
                handles=handles,
                loc='upper right',
                bbox_to_anchor=(0.995, header_y),
                ncol=len(handles),
                frameon=False,
                fontsize=12,
                handlelength=2.9,
                columnspacing=0.95,
                handletextpad=0.5,
                borderaxespad=0.0,
            )
        else:
            fig.legend(
                handles=handles,
                loc='upper center',
                bbox_to_anchor=(0.5, header_y),
                ncol=len(handles),
                frameon=False,
                fontsize=12,
                handlelength=2.9,
                columnspacing=0.95,
                handletextpad=0.5,
                borderaxespad=0.0,
            )

        box_style = dict(boxstyle='round,pad=0.28', facecolor='white', edgecolor='0.6', linewidth=0.8, alpha=0.96)
        ax0.text(
            0.5, 0.055,
            rf'$\mathrm{{S/N}}[y] = {snr0:.0f}\ /\ {snr0_db:.1f}\,\mathrm{{dB}}$',
            transform=ax0.transAxes, ha='center', va='bottom', fontsize=11.5, bbox=box_style,
        )
        ax1_bottom.text(
            0.5, 0.055,
            rf'$\mathrm{{S/N}}[\mu] = {snr:.0f}\ /\ {snr_db:.1f}\,\mathrm{{dB}}$',
            transform=ax1_bottom.transAxes, ha='center', va='bottom', fontsize=11.5, bbox=box_style,
        )
        ax2_bottom.text(
            0.5, 0.055,
            rf'$\mathrm{{S/N}}[\mu_{{\mathrm{{Ca\,II}}}}] = {ca_snr:.0f}\ /\ {ca_snr_db:.1f}\,\mathrm{{dB}}$',
            transform=ax2_bottom.transAxes, ha='center', va='bottom', fontsize=11.5, bbox=box_style,
        )

        fig.subplots_adjust(left=0.055, right=0.985, top=0.946, bottom=0.105, wspace=0.006, hspace=0.012)
        return fig

    def plot_snr_improve(self, N=10000, num_levels= 20, start_idx=3, bnd = [1, 10]):
        snr0 = self.d.snr_no_mask[:N]
        snr1 = self.d.snr[:N] 
        snr01 = np.vstack([snr0, snr1])
        kde = gaussian_kde(snr01)
        z = kde(snr01)
        x_min, x_max = snr0.min().item(), snr0.max().item()
        y_min, y_max = snr1.min().item(), snr1.max().item()
        X, Y = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = kde(positions).reshape(X.shape)
        levels = np.linspace(Z.min(), Z.max(), num_levels)
        mask = (z < levels[start_idx])

        fig, ax = plt.subplots(figsize=(5,4), dpi=self.dpi, facecolor='w')
        cs = ax.contourf(X, Y, Z, levels=levels[start_idx:], cmap='hot')  # , norm=norm

        # ax.contour(X, Y, Z,
        #    levels=np.percentile(Z, [50, 70, 85, 95, 98, 99]),  # 少量主等高线
        #    colors='black', linewidths=0.8)
        # cs = ax.contour(X, Y, Z, levels=levels[start_idx:], cmap='hot',)  
        cbar = fig.colorbar(cs, ax=ax)
        cbar.locator = ticker.MaxNLocator(nbins=6)
        cbar.set_label("Density")
        ax.scatter(snr0[mask], snr1[mask], s=0.5, c='k',)
        _=ax.set(ylim=(0, None), xlabel="Initial (Noisy) S/N ", ylabel="Final (Denoised) S/N")       

        fig1, ax1 = plt.subplots(figsize=(5,4), dpi=self.dpi, facecolor='w')
        # cs = ax1.contour(X, Y, Z, levels=levels[start_idx:], cmap='hot')  # 去掉最外圈
        cs1 = ax1.contourf(X, Y, Z, levels=levels[start_idx:], cmap='hot')  # , norm=norm
        cbar1 = fig1.colorbar(cs1, ax=ax1)
        cbar1.locator = ticker.MaxNLocator(nbins=6)
        cbar1.set_label("Density")
        
        ax1.scatter(snr0[mask], snr1[mask], s=0.5, c='k')
        ax1.plot(bnd, bnd, 'k:', label='Noisy')
        major_ticks = [1, 2, 3, 4, 5, 10]  # 可以按需增加
        _=ax1.set(xlabel='Initial (Noisy) S/N (log scale)', xscale='log', yscale='log', ylim=(1,None), ylabel='Final (Denoised) S/N (log scale)', xticks=major_ticks, xticklabels=[str(m) for m in major_ticks])
        return fig, fig1

    def get_equivalent_width(self, N=100):
        z = self.d.z[:N]
        df_flux = get_equivalent_width(self.wave, self.d.flux[:N], z)
        df_noisy =  get_equivalent_width(self.wave, self.d.noisy[:N], z)
        df_denoised = get_equivalent_width(self.wave, self.d.denoised[:N], z)
        return df_flux, df_noisy, df_denoised


    def get_equivalent_width_error_dict(self, N=1000, bins=None):
        self.index_columns = ["TiO_4", "Ca1_LB13", "Ca2_LB13", "Ca3_LB13"]
        if bins is None: bins = np.array([0, 2, 4, 6, 8, np.inf])
        snr_values = self.d.snr_no_mask[:N]
        df_flux, df_noisy, df_denoised = self.get_equivalent_width(N=N)
        error_data = []
        self.eq_bins = []
        self.rel_err_dict = {index: {f"S/N < {int(self.eq_bins[i])}": None for i in range(len(self.eq_bins)-1)} for index in self.index_columns}
        self.rel_err_mean_dict = {index: {'noisy': [], 'denoised': []} for index in self.index_columns}
        self.rel_err_std_dict = {index: {'noisy': [], 'denoised': []} for index in self.index_columns}
        for index_name in self.index_columns:
            rel_err0 = (df_noisy[index_name] - df_flux[index_name]) / df_flux[index_name]
            rel_err1 = (df_denoised[index_name] - df_flux[index_name]) / df_flux[index_name]
            for i in range(len(bins) - 1):
                snr_min, snr_max = bins[i], bins[i+1]
                idx = np.where((snr_values >= snr_min) & (snr_values < snr_max))[0]
                data_noisy, data_denoised = rel_err0[idx], rel_err1[idx]
                if len(data_noisy) == 0: break
                self.eq_bins.append(snr_min)
                self.rel_err_dict[index_name][f"S/N < {int(bins[i+1])}"] = (data_noisy, data_denoised)
                noisy_mean = data_noisy.mean()
                denoised_mean = data_denoised.mean()
                noisy_std = data_noisy.std()
                denoised_std = data_denoised.std()

                self.rel_err_mean_dict[index_name]['noisy'].append(noisy_mean)
                self.rel_err_mean_dict[index_name]['denoised'].append(denoised_mean)
                self.rel_err_std_dict[index_name]['noisy'].append(noisy_std)
                self.rel_err_std_dict[index_name]['denoised'].append(denoised_std)

                for err in data_denoised:
                    error_data.append({'index_name': index_name, 'snr_bin': f"{int(bins[i])} - {int(bins[i+1])}", 'type': 'denoised', 'bias': err, 'mean': denoised_mean, 'std': denoised_std})
                for err in data_noisy:
                    error_data.append({'index_name': index_name, 'snr_bin': f"{int(bins[i])} - {int(bins[i+1])}", 'type': 'noisy', 'bias': err, 'mean': noisy_mean, 'std': noisy_std})
        self.df_equivalent_width = pd.DataFrame(error_data)
        self.eq_N = N
        self.eq_bins = self.eq_bins[:len(self.index_columns)]

    def plot_equivalent_width(self, plot_snr_bin='0 - 2', plot_name="Ca2_LB13"):
        if self.df_equivalent_width is None:
            self.get_equivalent_width_error_dict()
        noisy_data = self.df_equivalent_width[(self.df_equivalent_width['index_name'] == plot_name) & (self.df_equivalent_width['snr_bin'] == plot_snr_bin) & (self.df_equivalent_width['type'] == 'noisy')]['bias'].values
        denoised_data = self.df_equivalent_width[(self.df_equivalent_width['index_name'] == plot_name) & (self.df_equivalent_width['snr_bin'] == plot_snr_bin) & (self.df_equivalent_width['type'] == 'denoised')]['bias'].values
        
        fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
        name = r'Ca II - 8542$\AA$'
        bin_name = f'S/N < {int(plot_snr_bin.split("-")[1].strip())}'
        SpecPlotter.plot_kde_distribution(ax, noisy_data, label="Noisy", color="k", bin_label=bin_name, index_name=name)
        SpecPlotter.plot_kde_distribution(ax, denoised_data, label="Denoised", color="blue", bin_label=bin_name, index_name=name)
        ax.axvline(0, color='r', lw=1, linestyle='-.')
        ax.set(xlabel="EW Relative Error Distribution", ylabel="Probability Density", xlim=(-2, 2))
        return fig

    def plot_equivalent_width_violin(self, plot_name="Ca2_LB13"):
        dff = self.df_equivalent_width[self.df_equivalent_width['index_name'] == plot_name].copy()
        dff['snr0_group1'] = dff['snr_bin']
        eq_stds_noisy = dff[dff['type'] == 'noisy'].groupby('snr_bin')['std'].first().to_dict()
        eq_stds_denoised = dff[dff['type'] == 'denoised'].groupby('snr_bin')['std'].first().to_dict()
        fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
        ax.axvline(0, linestyle='-', color='r', alpha=1, lw=0.5)
        name = r'Ca II - 8542$\AA$'
        ax.text(0.5, 0.98, name, transform=plt.gca().transAxes, ha='center', va='top', fontsize=12)
        sns.violinplot(x='bias', y='snr0_group1', hue='type', data=dff,split=True, gap=0.05, inner='quart', palette=['b','gray'], hue_order=['denoised', 'noisy'],linecolor='k', linewidth=0.5, ax=ax)
        ax.set(ylabel='Initial (Noisy) S/N Bin', xlabel='Equivalent Width Relative Error Distribution', xlim=(-1, 1))
        yticks = plt.gca().get_yticks()
        for i, y in enumerate(yticks[:5]):  # Assuming first 5 bins correspond
            bin_name = dff['snr_bin'].unique()[i]
            ax.text(-0.95, y - 0.08, f'σ = {eq_stds_denoised.get(bin_name, 0):.1g}', ha='left', va='center', fontsize=7, color='b')
            ax.text(-0.95, y + 0.12, f'σ = {eq_stds_noisy.get(bin_name, 0):.1g}', ha='left', va='center', fontsize=7, color='k')
        ax.legend(loc='upper right')
        return fig

    def plot_ew_std_comparison(self, index_columns=None):
        if index_columns is None: index_columns = self.index_columns
        bins = self.eq_bins
        color = ['k', 'b', 'r', 'g']
        marker = ['*', 'X', '^', 'o']
        fig, ax = plt.subplots(1, figsize=(5, 4), dpi=200)
        for i, index_name in enumerate(index_columns):
            std_denoised = self.rel_err_std_dict[index_name]['denoised']
            std_noisy = self.rel_err_std_dict[index_name]['noisy']
            ax.plot(bins, std_denoised, marker=marker[i], linestyle='-', c=color[i], label=f'{index_name[:3]} denoised', ms=5, lw=1)
            ax.plot(bins, std_noisy, marker=marker[i], linestyle=':', c=color[i], label=f'{index_name[:3]} noisy', ms=5, lw=1)

        ax.set(xlabel='Initial (Noisy) S/N', ylabel='Standard deviation of EW relative error', yscale='log', xticks=bins, xticklabels=[f'{m:.0f} - {m+2:.0f}' for m in bins])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.,frameon=True, ncol=1)
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        return fig


    @staticmethod
    def add_stats_text(ax, data, label="Denoised"):
        """Add mean, std, skew stats box to ax."""
        mean_val, std_val, skew_val = np.mean(data), np.std(data), skew(data)
        stats_text = (f'{label}:\n μ={mean_val:.1g}\n σ={std_val:.1g}\n skew={skew_val:.1g}')
        x_pos = 0.95
        y_pos = 0.95 if label == 'Denoised' else 0.7
        color = 'b' if label == 'Denoised' else 'k'
        ax.text(x_pos, y_pos, stats_text, transform=ax.transAxes, verticalalignment='top', horizontalalignment='right', fontsize=10,color=color,
                bbox=dict(boxstyle='round', edgecolor=color, facecolor='white', alpha=0.8))

    @staticmethod
    def plot_kde_distribution(ax, data, label="", color="blue", bin_label="", index_name=""):
        data = np.asarray(data, dtype=np.float64)
        data = data[~np.isnan(data)]
        kde = gaussian_kde(data)
        x_vals = np.linspace(np.quantile(data, 0.001), np.quantile(data,0.999), 200)
        pdf_vals = kde(x_vals)
        ax.plot(x_vals, pdf_vals, c=color, label=label)
        ax.hist(data, bins=50, density=True, alpha=0.2, color=color)
        SpecPlotter.add_stats_text(ax, data, label=label)
        ax.text(0.05, 0.95, index_name, transform=plt.gca().transAxes, ha='left', va='top', fontsize=12)
        ax.text(0.05, 0.85, bin_label, transform=plt.gca().transAxes, ha='left', va='top', fontsize=12)
