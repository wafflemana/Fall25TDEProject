import numpy as np
import matplotlib.pyplot as plt

#constants
ccc = 3.00 * 10**10      # speed of light [cm/s]
sbc = 1.38 * 10**(-16)   # k_B (Boltzmann constant) [erg/K]
hhh = 6.625 * 10**(-27)  # Planck's constant [erg*s]
CCC = 5.879 * 10**10     #Wein displacement law proportionality constant [Hz/K]
BHmass = 10**17 * (1.989 * 10**33) #Black hole mass [g]
GGG =  6.67 * 10**(-8)   #Gravitational constant [cm^3/g s^2]
rrr_g = (GGG * BHmass) / ccc**2 #Gravitational radius of the black hole

#load spectrum data
spec_file = np.loadtxt(r"C:\Users\Aviyel\PycharmProjects\Fall25TDEProject\models\m7am09_M22\3D\Spectra_150+\spec0150.dat")

#frequencies
nus = spec_file[:, 0]
uv = (nus >= 7.5e14) & (nus <= 3e16)
nus_uv = nus[uv]
nus_max_uv = nus_uv.max()

#black hole parameters
scr = (2 * GGG * BHmass) / (ccc**2) #Schwartzchild radius
r_min = scr / 10
r_max = rrr_g * 3000
rrr = np.logspace(np.log10(scr / 10), np.log10(rrr_g * 3000), 1000)

tbb = nus_max_uv / CCC #black body max temperature in K
t_min =  hhh * nus_max_uv / (sbc * 700)
t_max = tbb * 100
ttt = np.logspace(np.log10(t_min), np.log10(t_max), 1000)

test1 = (hhh * nus_max_uv) / (sbc * t_min)
nuLnusbb_min = nus_max_uv * 4 * np.pi**2 * r_max**2 * (2 * hhh * nus_max_uv**3) / (ccc**2) / (np.exp((hhh * nus_max_uv) / (sbc * t_min)) - 1)

print(f"Radius minimum = {r_min:.3e} cm")
print(f"Radius maximum = {r_max:.3e} cm")
print(f"Temperature minimum = {t_min:.3e} K")
print(f"Temperature maximum = {t_max:.3e} K")
print(f"Minimum Black Body nuLnu value = {nuLnusbb_min:.3e}")
print()

#black body model function, expected values
def nuLnusbb(nus_uv, r, T):
    bnu = (2 * hhh * nus_uv**3) / (ccc**2) / (np.exp((hhh * nus_uv) / (sbc * T)) - 1)
    lnu = 4 * np.pi**2 * r**2 * bnu
    return nus_uv * lnu

#observed values
nuLnusavg = spec_file[uv, 2]

#chi_square test
def chi_square(nuLnusbb, nuLnusavg):
    observed = nuLnusavg
    expected = nuLnusbb
    valid = expected > 1e-8
    return np.sum((observed[valid] - expected[valid])**2/ expected[valid])

# grid search
best_chi = np.inf
best_r = None
best_T = None

for r in rrr:
    for T in ttt:
        expected = nuLnusbb(nus_uv, r, T)
        chi = chi_square(expected, nuLnusavg)
        if chi < best_chi:
            best_chi = chi
            best_r = r
            best_T = T

print("Best fit parameters:")
print(f"Radius = {best_r:.3e} cm")
print(f"Temperature = {best_T:.3e} K")
print(f"Chi-square = {best_chi:.3e}")
