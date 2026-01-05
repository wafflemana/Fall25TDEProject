import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob

#directories
spec_path = r"/models/m7am09_M22/3D/Spectra_150+"
chisquare_path = r"/models/m7am09_M22/3D/chisquareresults\chisquare_results.csv"
save_path = r"C:\Users\Aviyel\PycharmProjects\Fall25TDEProject\models\m7am09_M22\chisquare comparison\comparisonplots"


#constants
ccc = 3.00e10      # speed of light [cm/s]
sbc = 1.38e-16     # Boltzmann constant [erg/K]
hhh = 6.625e-27    # Planck's constant [erg*s]
CCC = 5.879e10     # Wein displacement constant [Hz/K]
BHmass = 1e7 * (1.989e33)  # Black hole mass [g]
GGG = 6.67e-8      # Gravitational constant [cm^3/g/s^2]

#observered data
spec_file = r"/models/m7am09_M22/3D/Spectra_150+/spec0241.dat"
spec_data = np.loadtxt(spec_file)
spec_filename = os.path.basename(spec_file)
spec_num = ''.join(filter(str.isdigit, spec_filename))
print(spec_num)
nus = spec_data[:, 0]
nuLnusavg = spec_data[:, 2]

#chi-square fitted data
chisquare_data = pd.read_csv(chisquare_path)
chisquare_specnumber = (int(spec_num) - 150)
print(chisquare_specnumber)

chisquare_rrr_optuv = chisquare_data.iloc[chisquare_specnumber,1]
chisquare_ttt_optuv = chisquare_data.iloc[chisquare_specnumber,2]
chisquare_rrr_softxray = chisquare_data.iloc[chisquare_specnumber,4]
chisquare_ttt_softxray = chisquare_data.iloc[chisquare_specnumber,5]

#bb cruves
def planck_nu(nu, T):
    x = (hhh * nu) / (sbc * T)
    # Prevent overflow by limiting x
    x = np.clip(x, 1e-5, 700)
    return (2 * hhh * nu**3 / ccc**2) / (np.exp(x) - 1)

def nuLnusbb(nus, R, T):
    bnu = planck_nu(nus, T)
    lnu = 4 * np.pi**2 * R**2 * bnu
    return nus * lnu

#numerical checks
optuv_curve = nuLnusbb(nus, chisquare_rrr_optuv, chisquare_ttt_optuv)
softxray_curve = nuLnusbb(nus, chisquare_rrr_softxray, chisquare_ttt_softxray)
print("OptUV curve min/max:", np.nanmin(optuv_curve), np.nanmax(optuv_curve))
print("Soft X-ray curve min/max:", np.nanmin(softxray_curve), np.nanmax(softxray_curve))
print("OptUV R =", chisquare_rrr_optuv)
print("OptUV T =", chisquare_ttt_optuv)
print("SoftX-ray R =", chisquare_rrr_softxray)
print("SoftX-ray T =", chisquare_ttt_softxray)

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
    # Convert wavelength (cm) to frequency (Hz)
    nu_max_band = ccc / lambda_min_cm
    nu_min_band = ccc / lambda_max_cm
    ax.axvspan(nu_min_band, nu_max_band, color=colors[band], alpha=0.3, zorder=0)
    x_center = 10 ** ((np.log10(nu_min_band) + np.log10(nu_max_band)) / 2)
    ax.text(x_center, 1.02, band, transform=ax.get_xaxis_transform(),
            ha='center', va='bottom', fontsize=12, color='black')

# plot luminosity lines
plt.loglog(nus, nuLnusavg, color='black', linewidth=2, label=r'$L_{\nu,\mathrm{avg}}$')
plt.loglog(nus, optuv_curve, color='blue', linewidth=2, label=r'$L_{\mathrm{bb,optUV}}$', ls='--')
plt.loglog(nus, softxray_curve, color='red', linewidth=2, label=r'$L_{\mathrm{bb,softX-ray}}$', ls='--')

# plot limits
nu_max = max(nuLnusavg)
plt.ylim(1e38, 1e46)
plt.xlim(min(nus))

'''
# plot peak
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
'''

# figure details
plt.xlabel(r'Frequency $\nu$ (Hz)')
plt.ylabel(r'$\nu L_{\nu} \, (\mathrm{erg}\ \mathrm{s}^{-1})$')
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.show()
