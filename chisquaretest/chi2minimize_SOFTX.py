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

def nuLnu_bb(nu, r, T):
    Bnu = (2 * hhh * nu**3) / ccc**2 / (np.exp(hhh * nu / (k_b * T)) - 1)
    Lnu = 4 * np.pi**2 * r**2 * Bnu
    return nu * Lnu

def chi_square(model, data):
    valid = (model > 0) & (data > 0)
    return np.sum((model[valid] - data[valid])**2 / model[valid])

rrr = np.logspace(np.log10(scr / 10), np.log10(rrr_g * 3000), 300)

models_base = r"C:\Users\Aviyel\PycharmProjects\Fall25TDEProject\models"
models = [d for d in os.listdir(models_base)
          if os.path.isdir(os.path.join(models_base, d))]

angles = {
    "0deg": 4,
    "15deg": 6,
    "30deg": 8,
    "45deg": 10
}

for MODEL in models:

    spec_dir = os.path.join(models_base, MODEL, "Spectra_150+")
    save_dir = os.path.join(models_base, MODEL, "chi2results", "softx")
    os.makedirs(save_dir, exist_ok=True)

    spec_files = sorted(glob.glob(os.path.join(spec_dir, "spec*.dat")))

    for spec_path in spec_files:

        spec_name = os.path.basename(spec_path)
        spec_num = ''.join(filter(str.isdigit, spec_name))

        data = np.loadtxt(spec_path)
        nus = data[:, 0]
        nuLnusavg = data[:, 2]

        softx = (nus >= 0.3 * 2.41799e17) & (nus <= 10 * 2.41799e17)
        if not np.any(softx):
            continue

        nus_x = nus[softx]

        out_path = os.path.join(
            save_dir,
            f"chi2results_{MODEL}_{spec_num}_softx.dat"
        )

        with open(out_path, "w") as f:

            header = "radius_rg "
            for ang in angles:
                header += f"T_{ang} chi2_{ang} "
            header += "nuLnusavg_last\n"

            f.write(header)

            for r in rrr:

                line = f"{r/rrr_g:12.5e} "

                for ang, col in angles.items():

                    nuLnus = data[:, col]
                    nuLnus_x = nuLnus[softx]

                    t_min = hhh * nus_x.max() / (k_b * 700)
                    t_max = (nus_x.max() / CCC) * 100
                    ttt = np.logspace(np.log10(t_min), np.log10(t_max), 300)

                    best_chi = np.inf
                    best_T = None

                    for T in ttt:
                        model = nuLnu_bb(nus_x, r, T)
                        chi = chi_square(model, nuLnus_x)
                        if chi < best_chi:
                            best_chi = chi
                            best_T = T

                    if best_T is None:
                        line += "nan nan "
                    else:
                        line += f"{best_T:12.5e} {best_chi:12.5e} "

                line += f"{nuLnusavg[softx][-1]:12.5e}\n"
                f.write(line)

        print(f"Saved: {out_path}")
