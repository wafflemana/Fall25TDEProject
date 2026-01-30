import numpy as np
import matplotlib.pyplot as plt

# Constants
ccc = 3.00 * 10**10      # speed of light [cm/s]
k_b = 1.38 * 10**(-16)   # Boltzmann constant [erg/K]
hhh = 6.625 * 10**(-27)  # Planck's constant [erg*s]
CCC = 5.879 * 10**10     # Wien displacement law proportionality constant [Hz/K]
BHmass = 10**7 * (1.989 * 10**33)  # Black hole mass [g]
GGG = 6.67 * 10**(-8)    # Gravitational constant [cm^3/g s^2]
rrr_g = (GGG * BHmass) / ccc**2           # gravitational radius
scr = (2 * GGG * BHmass) / (ccc**2)       # Schwarzschild radius
L_edd = 1.26e38 * (BHmass / 1.989e33)      # Eddington luminosity

# Load spectrum data
spec_file = np.loadtxt(r"C:\Users\Aviyel\PycharmProjects\Fall25TDEProject\models\m7a0.9_M22\Spectra_150+\spec0250.dat")
nus = spec_file[:, 0]

#data from spec_file
optuv = (nus >= 4e14) & (nus <= 1.5e15)
nus_optuv = nus[optuv]
nus_max_optuv = nus_optuv.max()
nuLnusavg_optuv = spec_file[optuv, 2]

#data ranges
r_min = scr / 10
r_max = rrr_g * 3000
rrr = np.logspace(np.log10(r_min), np.log10(r_max), 1000)

tbb_optuv = nus_max_optuv / CCC
t_min_optuv = hhh * nus_max_optuv / (k_b * 700)
t_max_optuv = tbb_optuv * 100
ttt_optuv = np.logspace(np.log10(t_min_optuv),np.log10(t_max_optuv),1000)

# Blackbody functions
def Lnusbb(nus, r, T):
    bnu = (2 * hhh * nus**3) / (ccc**2) / (np.exp((hhh * nus) / (k_b * T)) - 1)
    return 4 * np.pi**2 * r**2 * bnu

def nuLnusbb(nus, Lnus):
    return nus * Lnus

# Chi-square
def chi_square(expected, observed):
    valid = (expected > 0) & (observed > 0)
    return np.sum((np.log10(expected[valid]) - np.log10(observed[valid]))**2)

# Chi-square minimization
chi_min_vs_r = []
best_T_vs_r = []

for r in rrr:
    chi_min_r = np.inf
    best_T_r = None

    for T in ttt_optuv:
        Lnu_peak = Lnusbb(nus_max_optuv, r, T)

        # Eddington limit
        if Lnu_peak > L_edd:
            continue

        model = nuLnusbb(nus_optuv, Lnusbb(nus_optuv, r, T))
        chi = chi_square(model, nuLnusavg_optuv)

        if chi < chi_min_r:
            chi_min_r = chi
            best_T_r = T

    chi_min_vs_r.append(chi_min_r)
    best_T_vs_r.append(best_T_r)

chi_min_vs_r = np.array(chi_min_vs_r)
best_T_vs_r = np.array(best_T_vs_r)

# Save results
file_path = (r"C:\Users\Aviyel\Documents\Research\Spring 2026\Chi v R graphs\m7a0.0_M22_250.txt")
data_out = np.column_stack((rrr, chi_min_vs_r))
np.savetxt(file_path, data_out, header="Radius[cm]   chi_min")
print(f"Array successfully saved to: {file_path}")

# Plot
plt.figure()
plt.loglog(rrr, chi_min_vs_r)
plt.xlabel("Radius [cm]")
plt.ylabel(r"Minimum $\chi^2$")
plt.title(r"Profiled $\chi^2$ vs Radius (Optical/UV), m7a0.9_M22 Model spec_file 250")
plt.grid(True, which="both")
plt.show()