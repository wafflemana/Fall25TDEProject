import numpy as np
import matplotlib.pyplot as plt
import os
import glob

#paths
data_dir = r"/models/m7am09_M22/3D/Spectra_151+"
save_dir = r"/models/chisquaretest/chisquareplots"
os.makedirs(save_dir, exist_ok=True)

#constants
ccc = 3.00 * 10**10 #speed of light in a vacuum in CGS units
sbc = 1.38 * 10**(-16)  #Stefan-Boltzmann constant in CGS units
hhh = 6.625 * 10**(-27)

#TDE specific constants
ttt = 2.5 * 10**4 #black body temperature in K
rrr = 3 * 10**14 #black body radius in cm

#EXTRACTED VALUES
spec_file = np.loadtxt(r"C:\Users\Aviyel\PycharmProjects\Fall25TDEProject\models\m7am09_M22\3D\Spectra_150+\spec0150.dat")

##wavelengths
nus = spec_file[:, 0]

## observed values
obsv_1 = spec_file[:,3]
obsv_2 = spec_file[:,5]
obsv_3 = spec_file[:,7]
obsv_4 = spec_file[:,9]

#COMPUTED VALUES

##expected values
def expected_value(nus, eps = 1e-12):
    bnu = ( (2 * hhh * nus**3) /(ccc**2) ) / ( (np.exp( (hhh * nus) / (sbc * ttt) ) ) - 1) #black body intensity formula
    val = 4 * np.pi**2 * rrr**2 * bnu #formula from R+L
    return val

# calculate expected values
expv_1 = expected_value(nus)
expv_2 = expected_value(nus)
expv_3 = expected_value(nus)
expv_4 = expected_value(nus)

#chi square time (finally)
def chi_square(observed, expected, cutoff = 150, eps = 1e-12):
    observed = observed[:cutoff]
    expected = expected[:cutoff]
    val = np.sum(((observed - expected) ** 2) / (expected + eps))
    return val

# compute chi-square values for each dataset
chi_sq_1 = chi_square(obsv_1, expv_1)
chi_sq_2 = chi_square(obsv_2, expv_2)
chi_sq_3 = chi_square(obsv_3, expv_3)
chi_sq_4 = chi_square(obsv_4, expv_4)


# print results
print(f"Chi-squared (θ = 0)     : {chi_sq_1}")
print(f"Chi-squared (θ = π/6)   : {chi_sq_2}")
print(f"Chi-squared (θ = π/3)   : {chi_sq_3}")
print(f"Chi-squared (θ = π/2)   : {chi_sq_4}")

def nuLnusbb(expected_value, nus):
    val = nus * expected_value(nus)
    return val

nuLnu_bb = nuLnusbb(expected_value, nus)

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


# Make the plot
plt.figure(figsize=(14,4))
plt.loglog(nus, nuLnu_bb, label=r'$\nu L_\nu$ (blackbody)')
plt.ylim(1e38, 1e46)
plt.xlim(min(nus), max(nus))

#EM bands
plt.axvspan(4e14, 8e14, color='lightblue', alpha=0.2, label='Optical')
plt.axvspan(8e14, 3e16, color='violet', alpha=0.2, label='UV')
plt.axvspan(3e16, 3e19, color='lightgreen', alpha=0.2, label='X-ray')

# Labels
plt.xlabel(r'Frequency $\nu$ (Hz)')
plt.ylabel(r'$\nu L_\nu$ (erg/s)')
plt.title(r'Blackbody Spectrum: $\nu L_\nu$ vs. $\nu$')
plt.grid(True, which="both", ls="--", alpha=0.5)

# Legend
plt.legend()
plt.tight_layout()

# Just show, no saving
plt.show()

