import numpy as np
import matplotlib.pyplot as plt

# Relevant constants used
hhh = 6.626 * 10**(-34)  # Planck's constant Js
ccc = 3 * 10**8          # speed of light m/s
kbb = 1.38 * 10**(-23)   # Boltzmann constant J/K
ttt = 300                # temperature in Kelvin

# Load spectrum data from the file
specdata = np.loadtxt("blackbody_sample_large.dat")

# Extract relevant columns
nus = specdata[:, 0]
I_nu = specdata[:, 2]  #would actually be nuLnuavgs in file

# Defining I_bb in terms of nu using Planck's law
I_bb = (2 * hhh * nus**3) / (ccc**2) / (np.exp(hhh * nus / (kbb * ttt)) - 1)

# Chi-Square fitting
chi_square = np.sum((I_nu - I_bb) ** 2 / I_bb)

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
plt.figure(figsize=(8, 6))
plt.plot(nus, I_nu, label=r'$I_\nu$', color='blue', linewidth=2)
plt.plot(nus, I_bb, label=r'$I_{bb}$', color='red', linestyle='--', linewidth=2)

# Labels and legend
plt.xlabel(r'Frequency $\nu$ (Hz)')
plt.ylabel(r'Intensity $I_\nu$ (W\,sr$^{-1}$\,m$^{-2}$\,Hz$^{-1}$)')
plt.title(r'\textbf{Observed vs Expected Spectrum}')
plt.legend(title=rf'$\chi^2 = {chi_square:.6e}$', loc='best')

# Show figure
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.tight_layout()
plt.show()
