import numpy as np
import matplotlib.pyplot as plt

# load spec data from file
specdata = np.loadtxt('spec0150.dat')

# extract relevant columns
nus = specdata[:, 0]
nuLnusavg = specdata[:, 2]
nuLnus1 = specdata[:, 4]
nuLnus2 = specdata[:, 6]
nuLnus3 = specdata[:, 8]
nuLnus4 = specdata[:, 10]

#chi squared calculations
valid = np.isfinite(nuLnusavg) & (nuLnusavg > 0)

chi_square1 = np.sum(((nuLnus1[valid] - nuLnusavg[valid]) ** 2) / nuLnusavg[valid])
chi_square2 = np.sum(((nuLnus2[valid] - nuLnusavg[valid]) ** 2) / nuLnusavg[valid])
chi_square3 = np.sum(((nuLnus3[valid] - nuLnusavg[valid]) ** 2) / nuLnusavg[valid])
chi_square4 = np.sum(((nuLnus4[valid] - nuLnusavg[valid]) ** 2) / nuLnusavg[valid])


# print data I wanna see
print(f'nuLnuvg_max = {max(nuLnusavg):.3e}')
print(f'Chi^2_1 = {chi_square1:.3e}')
print(f'Chi^2_2 = {chi_square2:.3e}')
print(f'Chi^2_3 = {chi_square3:.3e}')
print(f'Chi^2_4 = {chi_square4:.3e}')

# render plot in LaTex
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

# plot
plt.figure(figsize=(14, 4))

# plot luminosity lines
plt.loglog(nus, nuLnusavg, color='black', linewidth=2, label=r'$L_{\nu avg}$')
plt.loglog(nus, nuLnus1, color='blue', linewidth=2, label=r'$L_{\nu 1}$', linestyle='--')
plt.loglog(nus, nuLnus2, color='red', linewidth=2, label=r'$L_{\nu 2}$', linestyle='--')
plt.loglog(nus, nuLnus3, color='purple', linewidth=2, label=r'$L_{\nu 3}$', linestyle='--')
plt.loglog(nus, nuLnus4, color='orange', linewidth=2, label=r'$L_{\nu 4}$', linestyle='--')


# plot limits, includes 8 orders of magnitude along with the peaks
nu_max = max(nuLnusavg)
plt.ylim(10**(np.log10(nu_max) - 8), 10**(np.log10(nu_max) + 1))
plt.xlim(min(nus))

# peak plotting
peak_index = np.argmax(nuLnusavg)
nu_peak = nus[peak_index]
nuLnu_peak = nuLnusavg[peak_index]

# Annotate the peak
plt.annotate(
    rf'$\nu_{{\mathrm{{max}}}} = {nu_peak:.2e}\ \mathrm{{Hz}}$',
    xy=(nu_peak, nuLnu_peak),
    xytext=(1.2 * nu_peak, 2 * nuLnu_peak),
    fontsize=12
)

# misc. lines
plt.axvline(x=nu_peak, color='grey', linestyle='--', linewidth=1.5, label=r'$\nu_{\mathrm{max}}$')

# labels and legend
plt.xlabel(r'Frequency $\nu$ (Hz)')
plt.ylabel(r'$\nu L_{\nu} \, (\mathrm{erg}\ \mathrm{s}^{-1})$')
plt.legend()

# Show figure
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.show()