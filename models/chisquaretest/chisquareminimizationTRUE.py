import numpy as np
import pandas as pd
import os
import glob

#constants
ccc = 3.00e10      # speed of light [cm/s]
sbc = 1.38e-16     # Boltzmann constant [erg/K]
hhh = 6.625e-27    # Planck's constant [erg*s]
CCC = 5.879e10     # Wein displacement constant [Hz/K]
BHmass = 1e7 * (1.989e33)  # Black hole mass [g]
GGG = 6.67e-8      # Gravitational constant [cm^3/g/s^2]
rrr_g = (GGG * BHmass) / ccc**2  # gravitational radius [cm]

#paths
base_dir = r"C:\Users\Aviyel\PycharmProjects\Fall25TDEProject\models\m7am09_M22\3D\Spectra_150+"
save_path = r"C:\Users\Aviyel\PycharmProjects\Fall25TDEProject\models\chisquaretest\chisquareresults\chisquare_results.csv"

#chi-square function
def chi_square(expected, observed):
    valid = expected > 1e-8
    return np.sum((observed[valid] - expected[valid])**2 / expected[valid])

# blackbody fucntion model
def nuLnusbb(nus, r, T):
    bnu = (2 * hhh * nus**3) / (ccc**2) / (np.exp((hhh * nus) / (sbc * T)) - 1)
    lnu = 4 * np.pi**2 * r**2 * bnu
    return nus * lnu

#black hole constants
scr = (2 * GGG * BHmass) / (ccc**2)
r_min = scr / 10
r_max = rrr_g * 3000
rrr = np.logspace(np.log10(r_min), np.log10(r_max), 200)  # 200 points for faster computation

#results array
results = []
#load spec files
spec_files = sorted(glob.glob(os.path.join(base_dir, "spec*.dat")))

#loop through spectrum files
for spec_path in spec_files:
    # pull data from spec files
    spec_filename = os.path.basename(spec_path)
    spec_num = ''.join(filter(str.isdigit, spec_filename))
    spec_data = np.loadtxt(spec_path)

    nus = spec_data[:, 0]

    #optical/UV range
    optuv = (nus >= 4e14) & (nus <= 3e16)
    nus_optuv = nus[optuv]
    if len(nus_optuv) == 0:
        print(f"No optical/UV data in {spec_filename}")
        continue
    nus_max_optuv = nus_optuv.max()
    nuLnusavg_optuv = spec_data[optuv, 2]

    #soft x-ray range
    softxraymin = 0.3 * 2.41799e17
    softxraymax = 10 * 2.41799e17
    softxray = (nus >= softxraymin) & (nus <= softxraymax)
    nus_softxray = nus[softxray]
    if len(nus_softxray) == 0:
        print(f"No soft X-ray data in {spec_filename}")
        continue
    nus_max_softxray = nus_softxray.max()
    nuLnusavg_softxray = spec_data[softxray, 2]

    #temperature range definition
    tbb_optuv = nus_max_optuv / CCC
    t_min_optuv = hhh * nus_max_optuv / (sbc * 700)
    t_max_optuv = tbb_optuv * 100
    tbb_softxray = nus_max_softxray / CCC
    t_min_softxray = hhh * nus_max_softxray / (sbc * 700)
    t_max_softxray = tbb_softxray * 100
    ttt_optuv = np.logspace(np.log10(t_min_softxray), np.log10(t_max_softxray), 200) #same as soft xray
    ttt_softxray = np.logspace(np.log10(t_min_softxray), np.log10(t_max_softxray), 200) #same as optUV

    best_chi = np.inf
    best_r = best_T = None
    for r in rrr:
        for T in ttt_optuv:
            expected = nuLnusbb(nus_optuv, r, T)
            observed = nuLnusavg_optuv
            chi = chi_square(expected, observed)
            if chi < best_chi:
                best_chi, best_r, best_T = chi, r, T

    best_chi_x = np.inf
    best_r_x = best_T_x = None
    for r in rrr:
        for T in ttt_softxray:
            expected_x = nuLnusbb(nus_softxray, r, T)
            observed_x = nuLnusavg_softxray
            chi_x = chi_square(expected_x, observed_x)
            if chi_x < best_chi_x:
                best_chi_x, best_r_x, best_T_x = chi_x, r, T

    #save the results for the file
    results.append([
        spec_num,
        best_r, best_T, best_chi,
        best_r_x, best_T_x, best_chi_x
    ])

    print(f"Processed file: {spec_path}: χ²(opt/uv)={best_chi:.3e}, χ²(x-ray)={best_chi_x:.3e}")

#save data to CSV
df = pd.DataFrame(
    results,
    columns=[
        "Spec_File", "OptUV_Radius", "OptUV_Temperature", "OptUV_ChiSquare",
        "SoftX_Radius", "SoftX_Temperature", "SoftX_ChiSquare"
    ]
)

os.makedirs(os.path.dirname(save_path), exist_ok=True)
df.to_csv(save_path, index=False)
print(f"\nResults saved to:\n{save_path}")
