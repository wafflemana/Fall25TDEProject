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


# print data i wanna see
print(f'nu_max = {max(nuLnusavg):.3e}')
print(f'Chi^2_1 = {chi_square1:.3e}')
print(f'Chi^2_2 = {chi_square2:.3e}')
print(f'Chi^2_3 = {chi_square3:.3e}')
print(f'Chi^2_4 = {chi_square4:.3e}')

# LaTeX text rendering settings
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

# Plot
plt.figure(figsize=(14, 4))

# plot luminosity lines
plt.loglog(nus, nuLnusavg, color='black', linewidth=2, label=r'$L_{\nu avg}$')
plt.loglog(nus, nuLnus1, color='blue', linewidth=2, label=r'$L_{\nu 1}$', linestyle='--')
plt.loglog(nus, nuLnus2, color='red', linewidth=2, label=r'$L_{\nu 2}$', linestyle='--')
plt.loglog(nus, nuLnus3, color='purple', linewidth=2, label=r'$L_{\nu 3}$', linestyle='--')
plt.loglog(nus, nuLnus4, color='orange', linewidth=2, label=r'$L_{\nu 4}$', linestyle='--')

# plot misc. lines
# plt.axhline(y=max(nuLnusavg), color='black', linestyle='--', linewidth=1.5, label=r'$\nu_{\mathrm{max}}$')

# plot limits
plt.ylim(1e30, 1e46)
plt.xlim(min(nus))

# labels and legend
plt.xlabel(r'Frequency $\nu$ (Hz)')
plt.ylabel(r'$\nu L_{\nu} \, (\mathrm{erg}\ \mathrm{s}^{-1})$')
plt.legend()

# Show figure
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.show()