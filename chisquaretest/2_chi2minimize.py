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

# Model loop
models_base = r"C:\Users\Aviyel\PycharmProjects\Fall25TDEProject\models"
models = [d for d in os.listdir(models_base) if os.path.isdir(os.path.join(models_base, d))]

for MODEL in models:
    spec_dir = os.path.join(models_base, MODEL, "Spectra_150+")
    save_dir = os.path.join(models_base, MODEL, "chi2results")
    os.makedirs(save_dir, exist_ok=True)

    spec_files = sorted(glob.glob(os.path.join(spec_dir, "spec*.dat")))
    if not spec_files:
        continue

    print(f"\nProcessing model: {MODEL}")

    for spec_path in spec_files:
        spec_name = os.path.basename(spec_path)
        spec_num = ''.join(filter(str.isdigit, spec_name))

        data = np.loadtxt(spec_path)
        nus = data[:, 0]
        nuLnus = data[:, 2]

        # Optical / UV
        optuv = (nus >= 4e14) & (nus <= 1.5e15)
        if not np.any(optuv):
            print(f"No OUV data in {spec_name}")
            continue

        nus_ouv = nus[optuv]
        nuLnus_ouv = nuLnus[optuv]

        optuv_peak = nus_ouv[np.argmax(nuLnus_ouv)]
        LmaxOUV = 2.0 * nuLnus_ouv[-1]

        t_min = hhh * optuv_peak / (k_b * 100)
        t_max = (optuv_peak / CCC) * 100
        ttt_ouv = np.logspace(np.log10(t_min), np.log10(t_max), 300)

        # Soft X-ray
        softx = (nus >= 0.3 * 2.41799e17) & (nus <= 10 * 2.41799e17)
        has_xray = np.any(softx)

        if has_xray:
            nus_x = nus[softx]
            nuLnus_x = nuLnus[softx]

            t_min_x = hhh * nus_x.max() / (k_b * 700)
            t_max_x = (nus_x.max() / CCC) * 100
            ttt_x = np.logspace(np.log10(t_min_x), np.log10(t_max_x), 300)

        # Output file
        out_path = os.path.join(
            save_dir,
            f"chi2results_{MODEL}_{spec_num}.txt"
        )

        with open(out_path, "w") as f:
            f.write(
                "radius(rg)  "
                "best_T_OUV(K) best_L_OUV(erg/s) best_L_OUV/L_OUV best_L_OUV/L_Edd chi2_OUV  "
                "best_T_X(K) best_L_X(erg/s) best_L_X/L_Edd chi2_X\n"
            )

            for r in rrr:
                # OUV
                best_chi_ouv = np.inf
                best_T_ouv = best_L_ouv = None

                for T in ttt_ouv:
                    model = nuLnu_bb(nus_ouv, r, T)
                    chi = chi_square(model, nuLnus_ouv)

                    if chi < best_chi_ouv:
                        best_chi_ouv = chi
                        best_T_ouv = T
                        best_L_ouv = Lmax_bb(r, T)

                if best_L_ouv is None:
                    continue
                if best_L_ouv > L_edd or best_L_ouv < LmaxOUV:
                    continue

                #Soft X
                best_T_x = best_L_x = best_chi_x = np.nan

                if has_xray:
                    best_chi_x = np.inf
                    for T in ttt_x:
                        model_x = nuLnu_bb(nus_x, r, T)
                        chi_x = chi_square(model_x, nuLnus_x)
                        if chi_x < best_chi_x:
                            best_chi_x = chi_x
                            best_T_x = T
                            best_L_x = Lmax_bb(r, T)

                f.write(
                    f"{r/rrr_g:12.5e} "
                    f"{best_T_ouv:12.5e} {best_L_ouv:12.5e} "
                    f"{best_L_ouv/LmaxOUV:12.5e} {best_L_ouv/L_edd:12.5e} {best_chi_ouv:12.5e} "
                    f"{best_T_x if has_xray else np.nan:12.5e} "
                    f"{best_L_x if has_xray else np.nan:12.5e} "
                    f"{(best_L_x/L_edd) if has_xray else np.nan:12.5e} "
                    f"{best_chi_x if has_xray else np.nan:12.5e}\n"
                )

        print(f"Saved: {out_path}")
