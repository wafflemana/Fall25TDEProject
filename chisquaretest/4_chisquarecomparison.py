import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob

#directories
spec_path = r"C:\Users\aviye\PycharmProjects\Fall25TDEProject\models\m7am0.9_M22\Spectra_150-450"
chisquare_path = r"C:\Users\aviye\PycharmProjects\Fall25TDEProject\models\m7am0.9_M22\chisquareresults\chisquare_results_m7am0.9_M22.csv"
save_path = r"C:\Users\aviye\PycharmProjects\Fall25TDEProject\models\m7am0.9_M22\chisquarecomparison\comparisonplots"

#constants
ccc = 3.00e10      # speed of light [cm/s]
sbc = 1.38e-16     # Boltzmann constant [erg/K]
hhh = 6.625e-27    # Planck's constant [erg*s]
CCC = 5.879e10     # Wein displacement constant [Hz/K]
BHmass = 1e7 * (1.989e33)  # Black hole mass [g]
GGG = 6.67e-8      # Gravitational constant [cm^3/g/s^2]

# load chi-square data
chisquare_data = pd.read_csv(chisquare_path)

#lbb functions
def planck_nu(nu, T):
    x = (hhh * nu) / (sbc * T)
    x = np.clip(x, 1e-5, 700)
    return (2 * hhh * nu**3 / ccc**2) / (np.exp(x) - 1)

def nuLnusbb(nus, R, T):
    bnu = planck_nu(nus, T)
    lnu = 4 * np.pi**2 * R**2 * bnu
    return nus * lnu

#Latex Rendering
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

#loop
spec_files = sorted(glob.glob(os.path.join(spec_path, "spec*.dat")))

for spec_file in spec_files:
    spec_filename = os.path.basename(spec_file)
    spec_num = int(''.join(filter(str.isdigit, spec_filename)))

    # Load spectrum
    spec_data = np.loadtxt(spec_file)
    nus = spec_data[:, 0]
    nuLnusavg = spec_data[:, 2]

    # Find matching chi-square row
    chisquare_specnumber = (spec_num - 150)
    chisquare_rrr_optuv = chisquare_data.iloc[chisquare_specnumber, 1]
    chisquare_ttt_optuv = chisquare_data.iloc[chisquare_specnumber, 2]
    chisquare_rrr_softxray = chisquare_data.iloc[chisquare_specnumber, 4]
    chisquare_ttt_softxray = chisquare_data.iloc[chisquare_specnumber, 5]

    # Generate blackbody curves
    optuv_curve = nuLnusbb(nus, chisquare_rrr_optuv, chisquare_ttt_optuv)
    softxray_curve = nuLnusbb(nus, chisquare_rrr_softxray, chisquare_ttt_softxray)

    #plot the figure
    plt.figure(figsize=(14, 4))
    ax = plt.gca()

    # EM bands
    bands = {
        "X-ray": (1e-9, 1e-6),
        "UV": (1e-6, 4e-5),
        "Visible": (4e-5, 7e-5),
    }
    colors = {
        "X-ray": "lightblue",
        "UV": "violet",
        "Visible": "lightgreen"
    }

    for band, (lambda_min_cm, lambda_max_cm) in bands.items():
        nu_max_band = ccc / lambda_min_cm
        nu_min_band = ccc / lambda_max_cm
        ax.axvspan(nu_min_band, nu_max_band, color=colors[band], alpha=0.3, zorder=0)
        x_center = 10 ** ((np.log10(nu_min_band) + np.log10(nu_max_band)) / 2)
        ax.text(x_center, 1.02, band, transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=12, color='black')

    plt.loglog(nus, nuLnusavg, color='black', linewidth=2, label=r'$L_{\nu,\mathrm{avg}}$')
    plt.loglog(nus, optuv_curve, color='blue', linewidth=2, label=r'$L_{\mathrm{bb,optUV}}$')
    plt.loglog(nus, softxray_curve, color='red', linewidth=2, label=r'$L_{\mathrm{bb,softX-ray}}$')

    plt.xlabel(r'Frequency $\nu$ (Hz)')
    plt.ylabel(r'$\nu L_{\nu} \, (\mathrm{erg}\ \mathrm{s}^{-1})$')
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.ylim(1e38, 1e46)
    plt.xlim(min(nus), max(nus))

    #plot limits
    nuLnumax = max(optuv_curve)
    plt.ylim(nuLnumax * (10 ** (-10)), nuLnumax * (10 ** 2))

    # Save plot
    save_filename = f"{spec_num:04d}chisquarecompare.png"
    save_fullpath = os.path.join(save_path, save_filename)
    plt.savefig(save_fullpath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {save_filename}")
