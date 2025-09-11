import numpy as np
import matplotlib.pyplot as plt
import os
import scipy as sp

#paths
data_dir = r"C:\Users\aviye\PycharmProjects\Fall25TDEProject\models\m7am09_M22\3D\Spectra_150+"
plot_dir = r"C:\Users\aviye\PycharmProjects\Fall25TDEProject\models\chisquaretest\lbbplots"
chi_dir = r"C:\Users\aviye\PycharmProjects\Fall25TDEProject\models\chisquaretest\chisquarevalues"
os.makedirs(plot_dir, exist_ok=True)

#constants
ccc = 3.00 * 10**10 #speed of light in a vacuum in CGS units
sbc = 1.38 * 10**(-16)  #Stefan-Boltzmann constant in CGS units
hhh = 6.625 * 10**(-27)

#TDE specific constants
ttt = 2.5 * 10**4 #black body temperature in K
rrr = 3 * 10**14 #black body radius in cm

spec_file = np.loadtxt(r"C:\Users\aviye\PycharmProjects\Fall25TDEProject\models\m7am09_M22\3D\Spectra_150+\spec0150.dat")

#wavelengths
nus = spec_file[:, 0]

## observed values
obsv_1 = spec_file[:, 4]
obsv_2 = spec_file[:, 6]
obsv_3 = spec_file[:, 8]
obsv_4 = spec_file[:, 10]

#COMPUTED VALUES (past Avi was SO STUPID so now PRESENT AVI (ME!!!) has to FIX IT

#
def Lumbb(nus, eps = 1e-12):
    bnu = ( (2 * hhh * nus**3) /(ccc**2) ) / ( (np.exp( (hhh * nus) / (sbc * ttt) ) ) - 1) #black body intensity formula
    val = 4 * np.pi**2 * rrr**2 * bnu #formula from R+L
    return val

def nuLnusbb(Lumbb, nus):
    val = nus * Lumbb(nus)
    return val

nuLnu_bb = nuLnusbb(Lumbb, nus)

# Actual chi-square test
chi_sq_1 = sp.stats.chisquare(obsv_1, nuLnu_bb)
chi_sq_2 = sp.stats.chisquare(obsv_2, nuLnu_bb)
chi_sq_3 = sp.stats.chisquare(obsv_3, nuLnu_bb)
chi_sq_4 = sp.stats.chisquare(obsv_4, nuLnu_bb)

#print chi square numbers (tryna see wsp)
print(f"Chi Square Average: {chi_sq_1}")
print(f"Chi Square 1: {chi_sq_2}")
print(f"Chi Square 2: {chi_sq_3}")
print(f"Chi Square 3: {chi_sq_4}")

#Plotting Time YAHOOO

# Latex rendering
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


# Make the plot
plt.figure(figsize=(14,4))
plt.loglog(nus, nuLnu_bb, label=r'$\nu L_\nu$ (blackbody)')
plt.ylim(1e38, 1e46)
plt.xlim(min(nus), max(nus))

# EM bands (no labels in legend)
plt.axvspan(4e14, 8e14, color='lightblue', alpha=0.2)
plt.axvspan(8e14, 3e16, color='violet', alpha=0.2)
plt.axvspan(3e16, 3e19, color='lightgreen', alpha=0.2)
ymax = 1e46  # top of y-axis
plt.text(5.5e14, ymax*1.1, "Optical", ha="center", fontsize=12)
plt.text(1e15,  ymax*1.1, "UV", ha="center", fontsize=12)
plt.text(1e18,  ymax*1.1, "X-ray", ha="center", fontsize=12)

# Legend and labels
plt.xlabel(r'Frequency $\nu$ (Hz)')
plt.ylabel(r'$\nu L_\nu$ (erg/s)')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
