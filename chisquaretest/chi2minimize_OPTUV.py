import numpy as np
import os
import glob

# Constants
ccc = 3.00e10
k_b = 1.38e-16
hhh = 6.625e-27
CCC = 5.879e10
GGG = 6.67e-8

BHmass = 1e7 * 1.989e33
rrr_g = (GGG * BHmass) / ccc**2
scr = (2 * GGG * BHmass) / ccc**2
L_edd = 1.26e38 * (BHmass / 1.989e33)

# Functions
def nuLnu_bb(nu, r, T):
    Bnu = (2 * hhh * nu**3) / ccc**2 / (np.exp(hhh * nu / (k_b * T)) - 1)
    Lnu = 4 * np.pi**2 * r**2 * Bnu
    return nu * Lnu

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

models_base = r"C:\Users\Aviyel\PycharmProjects\Fall25TDEProject\models"
models = [d for d in os.listdir(models_base)
          if os.path.isdir(os.path.join(models_base, d))]

# Viewing angle mapping
angles = {
    "0deg": 4,
    "15deg": 6,
    "30deg": 8,
    "45deg": 10
}

for MODEL in models:

    spec_dir = os.path.join(models_base, MODEL, "Spectra_150+")
    save_dir = os.path.join(models_base, MODEL, "chi2results", "optuv")
    os.makedirs(save_dir, exist_ok=True)

    spec_files = sorted(glob.glob(os.path.join(spec_dir, "spec*.dat")))

    for spec_path in spec_files:

        spec_name = os.path.basename(spec_path)
        spec_num = ''.join(filter(str.isdigit, spec_name))

        data = np.loadtxt(spec_path)
        nus = data[:, 0]
        nuLnusavg = data[:, 2]

        # Optical / UV mask
        optuv = (nus >= 4e14) & (nus <= 1.5e15)
        if not np.any(optuv):
            continue

        out_path = os.path.join(
            save_dir,
            f"chi2results_{MODEL}_{spec_num}_optuv.dat"
        )

        with open(out_path, "w") as f:

            header = "radius_rg "
            for ang in angles:
                header += f"T_{ang} L_{ang} L_over_Edd_{ang} chi2_{ang} "
            header += "nuLnusavg_last\n"

            f.write(header)

            for r in rrr:

                line = f"{r / rrr_g:12.5e} "
                valid_row = True  # Track if row should be written

                for ang, col in angles.items():

                    nuLnus = data[:, col]
                    nus_ouv = nus[optuv]
                    nuLnus_ouv = nuLnus[optuv]

                    optuv_peak = nus_ouv[np.argmax(nuLnus_ouv)]
                    Lmin = 2.0 * nuLnus_ouv[-1]

                    t_min = hhh * optuv_peak / (k_b * 100)
                    t_max = (optuv_peak / CCC) * 100
                    ttt = np.logspace(np.log10(t_min), np.log10(t_max), 300)

                    best_chi = np.inf
                    best_T = None
                    best_L = None

                    for T in ttt:
                        model = nuLnu_bb(nus_ouv, r, T)
                        chi = chi_square(model, nuLnus_ouv)
                        if chi < best_chi:
                            best_chi = chi
                            best_T = T
                            best_L = Lmax_bb(r, T)

                    # Reject invalid solutions
                    if best_L is None or best_L > L_edd or best_L < Lmin:
                        valid_row = False
                        break

                    line += (
                        f"{best_T:12.5e} "
                        f"{best_L:12.5e} "
                        f"{(best_L / L_edd):12.5e} "
                        f"{best_chi:12.5e} "
                    )

                # Only write row if ALL angles were valid
                if valid_row:
                    line += f"{nuLnusavg[optuv][-1]:12.5e}\n"
                    f.write(line)

        print(f"Saved: {out_path}")
