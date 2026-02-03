import numpy as np
import matplotlib.pyplot as plt

# Constants
ccc = 3.00e10           # speed of light [cm/s]
k_b = 1.38e-16          # Boltzmann constant [erg/K]
hhh = 6.625e-27         # Planck constant [erg*s]
CCC = 5.879e10          # Wien constant [Hz/K]
GGG = 6.67e-8           # gravitational constant [cm^3/g/s^2]
BHmass = 1e7 * 1.989e33
rrr_g = (GGG * BHmass) / ccc**2
scr = (2 * GGG * BHmass) / ccc**2
L_edd = 1.26e38 * (BHmass / 1.989e33)


# Load spectrum
spec_file = np.loadtxt(r"/models/m7am0.9_M22/Spectra_150+\spec0250.dat")
nus = spec_file[:, 0]
nuLnus = spec_file[:, 2]
Lnus = spec_file[:, 1]

# Optical/UV selection
optuv = (nus >= 4e14) & (nus <= 1.5e15)
nus_optuv = nus[optuv]
nuLnus_optuv = nuLnus[optuv]
Lnus_optuv = Lnus[optuv]

# Peak quantities from KORAL spectrum
optuv_peak = nus_optuv[np.argmax(nuLnus_optuv)]
Lmin_OUV = 2.0 * nuLnus_optuv[-1]

# Radius & temperature grids
r_min = scr / 10
r_max = rrr_g * 3000
rrr = np.logspace(np.log10(r_min), np.log10(r_max), 300)

T_bb_peak = optuv_peak / CCC
t_min = hhh * optuv_peak / (k_b * 100)
t_max = T_bb_peak * 100
ttt = np.logspace(np.log10(t_min), np.log10(t_max), 300)

print(t_min)
print(t_max)

# functions
def Lnus_bb(nu, r, T):
    Bnu = (2 * hhh * nu**3) / (ccc**2) / (np.exp((hhh * nu) / (k_b * T)) - 1)
    return 4 * np.pi**2 * r**2 * Bnu

def nuLnu_bb(nu, r, T):
    return nu * Lnus_bb(nu, r, T)

def chi_square(model, data):
    valid = (model > 0) & (data > 0)
    return np.sum((model[valid] - data[valid])**2 / model[valid])

def Lmax_bb(r, T):
    nu_max = 5.88e10 * T  # Wien peak
    Bnu = (2 * hhh * nu_max**3) / (ccc**2) / (np.exp((hhh * nu_max) / (k_b * T)) - 1)
    Lnu = 4 * np.pi**2 * r**2 * Bnu
    return nu_max * Lnu

# Full (R,T) chi^2 map
results = []

for r in rrr:
    best_chi_local = np.inf
    best_T_local = None
    best_L_local = None

    for T in ttt:
        model = nuLnu_bb(nus_optuv, r, T)
        chi2 = chi_square(model, nuLnus_optuv)

        if chi2 < best_chi_local:
            best_chi_local = chi2
            best_T_local = T
            best_L_local = Lmax_bb(r, T)

    # Apply luminosity constraints AFTER minimization
    if best_L_local is None:
        continue
    if best_L_local > L_edd:
        continue
    if best_L_local < Lmin_OUV:
        continue

    results.append([r, best_T_local, best_chi_local])

results = np.array(results)

# Save output
out_file = (
    r"C:\Users\Aviyel\Documents\Research\Spring 2026\chivr_graphs_3D_v2"
    r"\m7am0.9_M22_250_R_T_chi2.txt")

np.savetxt(out_file, results, header="Radius[cm]   Temperature[K]   chi2")
print(f"Saved {len(results)} (R,T,chi^2) points to:\n{out_file}")

# 3D chi^2 visualization
R_vals = results[:, 0]
T_vals = results[:, 1]
chi2_vals = results[:, 2]

# Convert to log space for visualization
logR = np.log10(R_vals)
logT = np.log10(T_vals)
logchi2 = np.log10(chi2_vals)
order = np.argsort(R_vals)
logR_s = logR[order]
logT_s = logT[order]
logchi2_s = logchi2[order]

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')
sc = ax.plot(
    logR_s,
    logT_s,
    logchi2_s,
    color="navy",
    linewidth=2.5,
    label="Best-fit χ² ridge"
)

# Identify global minimum
idx_min = np.argmin(chi2_vals)
ax.scatter(
    logR[idx_min],
    logT[idx_min],
    logchi2[idx_min],
    color='red',
    s=60,
    label='Global minimum'
)

R_rg = R_vals[idx_min] / rrr_g
T_min = T_vals[idx_min]
label = (
    r"$R = {:.2f}\,r_g$" "\n"
    r"$T = {:.2e}\,\mathrm{{K}}$"
).format(R_rg, T_min)

ax.text(
    logR[idx_min],
    logT[idx_min],
    logchi2[idx_min],
    label,
    color='red',
    fontsize=10,
    ha='left',
    va='bottom'
)
ax.legend()

ax.set_xlabel(r'$\log_{10}(R\,[\mathrm{cm}])$')
ax.set_ylabel(r'$\log_{10}(T\,[\mathrm{K}])$')
ax.set_zlabel(r'$\log_{10}(\chi^2)$')


ax.set_title(r'3D $\chi^2(R,T)$ Surface (OUV, BB Fits) m7am0.9_M22 spec0250')

plt.tight_layout()
plt.show()

print(logT[idx_min], 10**logR[idx_min] / rrr_g)
