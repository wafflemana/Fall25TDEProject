import numpy as np
import matplotlib.pyplot as plt
import os

# constants (CGS units)
ccc = 3.00 * 10**10      # speed of light [cm/s]
sbc = 1.38 * 10**(-16)   # k_B (Boltzmann constant) [erg/K]
hhh = 6.625 * 10**(-27)  # Planck's constant [erg*s]
CCC = 5.879 * 10**10 #Wein displacement law proportionality constant [Hz/K]


# load spectrum
spec_file = np.loadtxt(r"C:\Users\Aviyel\PycharmProjects\Fall25TDEProject\models\m7am09_M22\3D\Spectra_150+\spec0150.dat")

# frequencies
nus = spec_file[:, 0]

# observed νLν values
obsv_avg = spec_file[:,2]
obsv_1 = spec_file[:, 4]
obsv_2 = spec_file[:, 6]
obsv_3 = spec_file[:, 8]
obsv_4 = spec_file[:, 10]

# Black Hole stuff specific constants
scr = 2.954 * 10**22 #schwartzchild radius [cm]
rrr = np.linspace(scr/10**10, scr, 10000)

def ttt(nus, CCC): #temperature range for black body using Wein Displacement law
    ttt = nus/CCC
    return ttt

# blackbody model function
def nuLnubb(nus, ttt, rrr):
    bnu = (2 * hhh * nus**3) / (ccc**2) / (np.exp((hhh * nus) / (sbc * ttt)) - 1)
    lnu = 4 * np.pi**2 * rrr**2 * bnu
    return nus * lnu

# define UV range
nus_uv = nus[20:122]

# blackbody model for UV range only
nuLnubb_uv = nuLnubb(nus_uv)

# observed values in UV range
obsvavg_uv = obsv_avg[20:122]
obsv1_uv = obsv_1[20:122]
obsv2_uv = obsv_2[20:122]
obsv3_uv = obsv_3[20:122]
obsv4_uv = obsv_4[20:122]

# chi-square
def chi_square(observed, expected, eps=1e-12):
    return np.sum((observed - expected) ** 2 / (expected + eps))

chi_sq_avg = chi_square(obsvavg_uv, nuLnubb_uv)
chi_sq_1 = chi_square(obsv1_uv, nuLnubb_uv)
chi_sq_2 = chi_square(obsv2_uv, nuLnubb_uv)
chi_sq_3 = chi_square(obsv3_uv, nuLnubb_uv)
chi_sq_4 = chi_square(obsv4_uv, nuLnubb_uv)

# print results
print(f"avg : {chi_sq_avg}")
print(f"θ = 0   : {chi_sq_1}")
print(f"θ = π/6 : {chi_sq_2}")
print(f"θ = π/3 : {chi_sq_3}")
print(f"θ = π/2 : {chi_sq_4}")