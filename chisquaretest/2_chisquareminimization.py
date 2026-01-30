import numpy as np
import pandas as pd
import os
import glob

# Constants
ccc = 3.00e10  #speed of light [cm/s]
k_b = 1.38e-16 #Boltzman constant
hhh = 6.625e-27 #Plank constant
CCC = 5.879e10  #Wein constant
GGG = 6.67e-8 #Gravitational constat
BHmass = 1e7 * 1.989e33

rrr_g = (GGG * BHmass) / ccc**2  #gravitational radius
scr = (2 * GGG * BHmass) / ccc**2 #Schwartzchild radius
L_edd = 1.26e38 * (BHmass / 1.989e33) #Eddington Luminosity

# Paths
base_dir = r"C:\Users\Aviyel\PycharmProjects\Fall25TDEProject\models\m7a0.9_M22\Spectra_150+"
save_dir = r"C:\Users\Aviyel\PycharmProjects\Fall25TDEProject\models\m7a0.9_M22\chisquareresults"
os.makedirs(save_dir, exist_ok=True)

# Functions
def nuLnu_bb(nus, r, T):
    Bnu = (2 * hhh * nus**3) / ccc**2 / (np.exp(hhh * nus / (k_b * T)) - 1)
    Lnu = 4 * np.pi**2 * r**2 * Bnu
    return nus * Lnu

def chi_square(model, data):
    valid = (model > 0) & (data > 0)
    return np.sum((model[valid] - data[valid])**2 / model[valid])

def Lmax_bb(r, T):
    nu_max = 5.88e10 * T
    Bnu = (2 * hhh * nu_max**3) / ccc**2 / (np.exp(hhh * nu_max / (k_b * T)) - 1)
    Lnu = 4 * np.pi**2 * r**2 * Bnu
    return nu_max * Lnu

# Radius grid
rrr = np.logspace(np.log10(scr / 10), np.log10(rrr_g * 3000), 300)

# Storage
results = []

# Loop over spectra
spec_files = sorted(glob.glob(os.path.join(base_dir, "spec*.dat")))

for spec_path in spec_files:
    spec_name = os.path.basename(spec_path)
    spec_num = ''.join(filter(str.isdigit, spec_name))
    spec = np.loadtxt(spec_path)

    nus = spec[:, 0]
    nuLnus = spec[:, 2]

    # Optical / UV
    optuv = (nus >= 4e14) & (nus <= 1.5e15)
    if not np.any(optuv):
        print(f"No OUV data in {spec_name}")
        continue

    nus_optuv = nus[optuv]
    nuLnus_optuv = nuLnus[optuv]

    optuv_peak = nus_optuv[np.argmax(nuLnus_optuv)]
    Lmin_OUV = 2.0 * nuLnus_optuv[-1]

    T_bb_peak = optuv_peak / CCC
    t_min = hhh * optuv_peak / (k_b * 100)
    t_max = T_bb_peak * 100
    ttt_optuv = np.logspace(np.log10(t_min), np.log10(t_max), 300)

    best_chi_ouv = np.inf
    best_r_ouv = best_T_ouv = None

    for r in rrr:
        best_chi_local = np.inf
        best_T_local = None
        best_L_local = None

        for T in ttt_optuv:
            model = nuLnu_bb(nus_optuv, r, T)
            chi = chi_square(model, nuLnus_optuv)

            if chi < best_chi_local:
                best_chi_local = chi
                best_T_local = T
                best_L_local = Lmax_bb(r, T)

        if best_L_local is None:
            continue
        if best_L_local > L_edd:
            continue
        if best_L_local < Lmin_OUV:
            continue

        if best_chi_local < best_chi_ouv:
            best_chi_ouv = best_chi_local
            best_r_ouv = r
            best_T_ouv = best_T_local

    # Soft X-ray
    softxray = (nus >= 0.3 * 2.41799e17) & (nus <= 10 * 2.41799e17)
    if np.any(softxray):
        nus_x = nus[softxray]
        nuLnus_x = nuLnus[softxray]

        tbb_x = nus_x.max() / CCC
        t_min_x = hhh * nus_x.max() / (k_b * 700)
        t_max_x = tbb_x * 100
        ttt_x = np.logspace(np.log10(t_min_x), np.log10(t_max_x), 300)

        best_chi_x = np.inf
        best_r_x = best_T_x = None

        for r in rrr:
            for T in ttt_x:
                model_x = nuLnu_bb(nus_x, r, T)
                chi_x = chi_square(model_x, nuLnus_x)
                if chi_x < best_chi_x:
                    best_chi_x = chi_x
                    best_r_x = r
                    best_T_x = T
    else:
        best_r_x = best_T_x = best_chi_x = None

    results.append([
        spec_num,
        best_r_ouv, best_T_ouv, best_chi_ouv,
        best_r_x, best_T_x, best_chi_x
    ])

    print(f"{spec_name}: χ²_OUV={best_chi_ouv:.3e}, χ²_X={best_chi_x}")

# Save CSV
df = pd.DataFrame(
    results,
    columns=[
        "Spec_File",
        "OptUV_Radius_cm", "OptUV_Temperature_K", "OptUV_ChiSquare",
        "SoftX_Radius_cm", "SoftX_Temperature_K", "SoftX_ChiSquare"
    ]
)

csv_path = os.path.join(save_dir, "chisquare_results_m70.9_M22.csv")
df.to_csv(csv_path, index=False)
print(f"\nSaved results to:\n{csv_path}")