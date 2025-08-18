import numpy as np
import matplotlib.pyplot as plt
import os
import glob

#constants
ccc = 299792458

# Paths
data_dir = r"C:\Users\Aviyel\PycharmProjects\Fall25TDEProject\models\m7am09_M22\3D\Spectra_150+"
save_dir = r"C:\Users\Aviyel\PycharmProjects\Fall25TDEProject\models\m7am09_M22\3D\lumoplots"
os.makedirs(save_dir, exist_ok=True)

# load spectra files
spec_files = sorted(glob.glob(os.path.join(data_dir, "spec*.dat")))

for spec_path in spec_files:
    #check if file is being processed
    print(f"Processing file: {spec_path}")

    #pull data from spec files
    spec_filename = os.path.basename(spec_path)
    spec_number = ''.join(filter(str.isdigit, spec_filename))
    specdata = np.loadtxt(spec_path)
    nus = specdata[:, 0]
    nuLnusavg = specdata[:, 2]
    nuLnus1 = specdata[:, 4]
    nuLnus2 = specdata[:, 6]
    nuLnus3 = specdata[:, 8]
    nuLnus4 = specdata[:, 10]
    '''
    # Chi-squared
    valid = np.isfinite(nuLnusavg) & (nuLnusavg > 0)
    chi_square1 = np.sum(((nuLnus1[valid] - nuLnusavg[valid]) ** 2) / nuLnusavg[valid])
    chi_square2 = np.sum(((nuLnus2[valid] - nuLnusavg[valid]) ** 2) / nuLnusavg[valid])
    chi_square3 = np.sum(((nuLnus3[valid] - nuLnusavg[valid]) ** 2) / nuLnusavg[valid])
    chi_square4 = np.sum(((nuLnus4[valid] - nuLnusavg[valid]) ** 2) / nuLnusavg[valid])

    print(f"{spec_filename}: Chi^2_1 = {chi_square1:.3e}, Chi^2_2 = {chi_square2:.3e}, "
          f"Chi^2_3 = {chi_square3:.3e}, Chi^2_4 = {chi_square4:.3e}")
    '''
    #Latex rendering
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 14,
        "font.size": 12,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12
    })

    plt.figure(figsize=(14, 4))
    ax = plt.gca()

    #EM bands
    bands = {
        "X-ray": (1e-11, 1e-8),
        "UV": (1e-8, 4e-7),
        "Visible": (4e-7, 7e-7)
    }
    colors = {
        "X-ray": "lightblue",
        "UV": "violet",
        "Visible": "lightgreen"
    }
    for band, (lambda_min, lambda_max) in bands.items():
        nu_max_band = ccc / lambda_min
        nu_min_band = ccc / lambda_max
        ax.axvspan(nu_min_band, nu_max_band, color=colors[band], alpha=0.3, zorder=0)
        x_center = 10 ** ((np.log10(nu_min_band) + np.log10(nu_max_band)) / 2)
        ax.text(x_center, 1.02, band, transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=12, color='black')

    # plot luminosity lines
    plt.loglog(nus, nuLnusavg, color='black', linewidth=2, label=r'$L_{\nu,\mathrm{avg}}$')
    plt.loglog(nus, nuLnus1, color='blue', linewidth=2, label=r'$L_{\nu,\,\theta=0}$')
    plt.loglog(nus, nuLnus2, color='red', linewidth=2, label=r'$L_{\nu,\,\theta=\pi/6}$')
    plt.loglog(nus, nuLnus3, color='purple', linewidth=2, label=r'$L_{\nu,\,\theta=\pi/3}$')
    plt.loglog(nus, nuLnus4, color='orange', linewidth=2, label=r'$L_{\nu,\,\theta=\pi/2}$')

    #plot limits
    nu_max = max(nuLnusavg)
    plt.ylim(1e38, 1e46)
    plt.xlim(min(nus))

    #plot peak
    peak_index = np.argmax(nuLnusavg)
    nu_peak = nus[peak_index]
    nuLnu_peak = nuLnusavg[peak_index]
    plt.annotate(
        rf'$\nu_{{\mathrm{{max}}}} = {nu_peak:.2e}\ \mathrm{{Hz}}$',
        xy=(nu_peak, nuLnu_peak),
        xytext=(1.2 * nu_peak, 2 * nuLnu_peak),
        fontsize=12
    )
    plt.axvline(x=nu_peak, color='grey', linestyle='--', linewidth=1.5, label=r'$\nu_{\mathrm{max}}$')

    #figure details
    plt.xlabel(r'Frequency $\nu$ (Hz)')
    plt.ylabel(r'$\nu L_{\nu} \, (\mathrm{erg}\ \mathrm{s}^{-1})$')
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)

    #save pngs
    save_path = os.path.join(save_dir, f"{spec_number}lumoplot.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()