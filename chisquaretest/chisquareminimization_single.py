import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#constants
ccc = 3.00 * 10**10      # speed of light [cm/s]
sbc = 1.38 * 10**(-16)   # k_B (Boltzmann constant) [erg/K]
hhh = 6.625 * 10**(-27)  # Planck's constant [erg*s]
CCC = 5.879 * 10**10     #Wein displacement law proportionality constant [Hz/K]
BHmass = 10**7 * (1.989 * 10**33) #Black hole mass [g]
GGG =  6.67 * 10**(-8)   #Gravitational constant [cm^3/g s^2]
rrr_g = (GGG * BHmass) / ccc**2 #Gravitational radius of the black hole

#load spectrum data
spec_file = np.loadtxt(r"/models/m7am09_M22/3D/Spectra_150+/spec0150.dat")

#frequencies
nus = spec_file[:, 0]
optuv = (nus >= 4e14) & (nus <= 3e16)
nus_optuv = nus[optuv]
nus_max_optuv = nus_optuv.max()

#black hole parameters (optical UV range)
scr = (2 * GGG * BHmass) / (ccc**2) #Schwartzchild radius
r_min = scr / 10
r_max = rrr_g * 3000
rrr = np.logspace(np.log10(scr / 10), np.log10(rrr_g * 3000), 1000)

tbb_optuv = nus_max_optuv / CCC #black body max temperature in K
t_min_optuv =  hhh * nus_max_optuv / (sbc * 700)
t_max_optuv = tbb_optuv * 100
ttt_optuv = np.logspace(np.log10(t_min_optuv), np.log10(t_max_optuv), 1000)

test1 = (hhh * nus_max_optuv) / (sbc * t_min_optuv)
nuLnusbb_min_optUV = nus_max_optuv * 4 * np.pi**2 * r_max**2 * (2 * hhh * nus_max_optuv**3) / (ccc**2) / (np.exp((hhh * nus_max_optuv) / (sbc * t_min_optuv)) - 1)

print(f"Radius minimum = {r_min:.3e} cm")
print(f"Radius maximum = {r_max:.3e} cm")
print(f"Temperature minimum Optical/UV = {t_min_optuv:.3e} K")
print(f"Temperature maximum Optical/UV = {t_max_optuv:.3e} K")
print(f"Minimum Black Body nuLnu value Optical/UV = {nuLnusbb_min_optUV:.3e}")
print()

#black body model function, expected values (optical UV)
def nuLnusbb_optuv(nus_uv, r, T):
    bnu = (2 * hhh * nus_optuv**3) / (ccc**2) / (np.exp((hhh * nus_optuv) / (sbc * T)) - 1)
    lnu = 4 * np.pi**2 * r**2 * bnu
    return nus_uv * lnu

#observed values (optical UV)
nuLnusavg_optuv = spec_file[optuv, 2]

#chi_square test (optical UV)
def chi_square(nuLnusbb_optuv, nuLnusavg_optuv):
    observed = nuLnusavg_optuv
    expected = nuLnusbb_optuv
    valid = expected > 1e-8
    return np.sum((observed[valid] - expected[valid])**2/ expected[valid])

# grid search
best_chi = np.inf
best_r = None
best_T = None

for r in rrr:
    for T in ttt_optuv:
        expected = nuLnusbb_optuv(nus_optuv, r, T)
        chi = chi_square(expected, nuLnusavg_optuv)
        if chi < best_chi:
            best_chi = chi
            best_r = r
            best_T = T

print("Best fit parameters (Optical/UV):")
print(f"Radius = {best_r:.3e} cm")
print(f"Temperature = {best_T:.3e} K")
print(f"Chi-square = {best_chi:.3e}")
print()

#frequencies (soft x-ray)
softxraymin = 0.3 * 2.41799e17
softxraymax = 10 * 2.41799e17
softxray = (nus >= softxraymin) & (nus <= softxraymax)
nus_softxray = nus[softxray]
nus_max_softxray = nus_softxray.max()

#black hole parameters (soft x-ray)
tbb_softxray = nus_max_softxray / CCC #black body max temperature in K
t_min_softxray =  hhh * nus_max_softxray / (sbc * 700)
t_max_softxray = tbb_softxray * 100
ttt_softxray = np.logspace(np.log10(t_min_softxray), np.log10(t_max_softxray), 1000)
nuLnusbb_min_softxray = nus_max_softxray * 4 * np.pi**2 * r_max**2 * (2 * hhh * nus_max_softxray**3) / (ccc**2) / (np.exp((hhh * nus_max_softxray) / (sbc * t_min_softxray)) - 1)

print(f"Temperature minimum soft x ray = {t_min_softxray:.3e} K")
print(f"Temperature maximum soft x ray = {t_max_softxray:.3e} K")
print(f"Minimum Black Body nuLnu value soft x ray = {nuLnusbb_min_softxray:.3e}")
print()

#black body model function (soft x-ray range)
def nuLnusbb(nus_softxray, r, T):
    bnu = (2 * hhh * nus_softxray**3) / (ccc**2) / (np.exp((hhh * nus_softxray) / (sbc * T)) - 1)
    lnu = 4 * np.pi**2 * r**2 * bnu
    return nus_softxray * lnu

# observed values (soft x-ray)
nuLnusavg_softxray = spec_file[softxray, 2]

#chi-square test (soft x-ray)
best_chi_x = np.inf
best_r_x = None
best_T_x = None

for r in rrr:
    for T in ttt_softxray:
        expected_x = nuLnusbb(nus_softxray, r, T)
        chi_x = chi_square(expected_x, nuLnusavg_softxray)
        if chi_x < best_chi_x:
            best_chi_x = chi_x
            best_r_x = r
            best_T_x = T

print("Best fit parameters (Soft X-ray):")
print(f"Radius = {best_r_x:.3e} cm")
print(f"Temperature = {best_T_x:.3e} K")
print(f"Chi-square = {best_chi_x:.3e}")


''' Grid Plotting 
chi_grid = np.zeros((len(rrr), len(ttt)))

for i, r in enumerate(rrr):
    for j, T in enumerate(ttt):
        expected = nuLnusbb(nus_optuv, r, T)
        chi_grid[i, j] = chi_square(expected, nuLnusavg_optuv)

# Make meshgrid for plotting
R, T = np.meshgrid(rrr, ttt, indexing="ij")  # ensure dimensions match chi_grid

#Actual 3D plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(np.log10(R), np.log10(T), np.log10(chi_grid),
                       cmap="viridis", edgecolor="none")

#axes labels and color
ax.set_xlabel("log10(Radius) [cm]")
ax.set_ylabel("log10(Temperature) [K]")
ax.set_zlabel("log10(Chi-square)")

fig.colorbar(surf, shrink=0.6, label="log10(Chi-square)")
plt.title("Chi-square Surface over Radius & Temperature")
plt.show()
'''