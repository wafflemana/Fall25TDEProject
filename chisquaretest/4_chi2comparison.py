import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# Paths (EDIT MODEL ONLY)
MODEL = "m7am0.9_M22"

base_dir = rf"C:\Users\Aviyel\PycharmProjects\Fall25TDEProject\models\{MODEL}"
spec_dir = os.path.join(base_dir, "Spectra_150+")
chi2_dir = os.path.join(base_dir, "chi2results")
save_dir = os.path.join(base_dir, "chisquarecomparison", "comparisonplots")
os.makedirs(save_dir, exist_ok=True)


# Constants
ccc = 3.00e10
k_b = 1.38e-16
hhh = 6.625e-27
CCC = 5.879e10
GGG = 6.67e-8

BHmass = 1e7 * 1.989e33
rrr_g = (GGG * BHmass) / ccc**2

# Blackbody functions
def planck_nu(nu, T):
    x = (hhh * nu) / (k_b * T)
    x = np.clip(x, 1e-5, 700)
    return (2 * hhh * nu**3 / ccc**2) / (np.exp(x) - 1)

def nuLnu_bb(nus, R, T):
    Bnu = planck_nu(nus, T)
    Lnu = 4 * np.pi**2 * R**2 * Bnu
    return nus * Lnu

# Plot style (LaTeX)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 14,
    "font.size": 12,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12
})

# Loop over spectra
spec_files = sorted(glob.glob(os.path.join(spec_dir, "spec*.dat")))

for spec_path in spec_files:
    spec_name = os.path.basename(spec_path)
    spec_num = ''.join(filter(str.isdigit, spec_name))

    chi2_path = os.path.join(
        chi2_dir,
        f"chi2results_{MODEL}_{spec_num}.txt"
    )

    if not os.path.exists(chi2_path):
        print(f"Missing chi² file for {spec_name}")
        continue

    # Load spectrum
    spec_data = np.loadtxt(spec_path)
    nus = spec_data[:, 0]
    nuLnus = spec_data[:, 2]

    # Load chi² results
    chi2_data = np.loadtxt(chi2_path, skiprows=1)

    # Columns:
    # 0 r_rg
    # 1 T_OUV
    # 2 L_OUV
    # 3 L/L_OUV
    # 4 L/L_Edd
    # 5 chi2_OUV
    # 6 T_X
    # 7 L_X
    # 8 L_X/L_Edd
    # 9 chi2_X

    # Best OUV fit
    idx_ouv = np.nanargmin(chi2_data[:, 5])
    r_ouv = chi2_data[idx_ouv, 0] * rrr_g
    T_ouv = chi2_data[idx_ouv, 1]

    # Best X-ray fit (if exists)
    if np.any(np.isfinite(chi2_data[:, 9])):
        idx_x = np.nanargmin(chi2_data[:, 9])
        r_x = chi2_data[idx_x, 0] * rrr_g
        T_x = chi2_data[idx_x, 6]
        has_xray = True
    else:
        has_xray = False

    # BB curves
    bb_ouv = nuLnu_bb(nus, r_ouv, T_ouv)
    bb_x = nuLnu_bb(nus, r_x, T_x) if has_xray else None

    # Build OUV envelope
    valid_ouv = np.isfinite(chi2_data[:, 5])

    bb_ouv_all = []
    for row in chi2_data[valid_ouv]:
        r_cm = row[0] * rrr_g
        T = row[1]
        bb_ouv_all.append(nuLnu_bb(nus, r_cm, T))

    bb_ouv_all = np.array(bb_ouv_all)
    bb_ouv_min = np.nanmin(bb_ouv_all, axis=0)
    bb_ouv_max = np.nanmax(bb_ouv_all, axis=0)

    # Build X-ray envelope
    valid_x = np.isfinite(chi2_data[:, 9])

    if np.any(valid_x):
        bb_x_all = []
        for row in chi2_data[valid_x]:
            r_cm = row[0] * rrr_g
            T = row[6]
            bb_x_all.append(nuLnu_bb(nus, r_cm, T))

        bb_x_all = np.array(bb_x_all)
        bb_x_min = np.nanmin(bb_x_all, axis=0)
        bb_x_max = np.nanmax(bb_x_all, axis=0)
        has_xray = True
    else:
        has_xray = False

    # Plot
    plt.figure(figsize=(14, 4))
    ax = plt.gca()

    # EM bands (unchanged)
    bands = {
        "X-ray": (1e-9, 1e-6),
        "UV": (1e-6, 4e-5),
        "Visible": (4e-5, 7e-5),
    }
    colors = {
        "X-ray": "lightblue",
        "UV": "violet",
        "Visible": "lightgreen"
    }

    for band, (lam_min, lam_max) in bands.items():
        nu_max = ccc / lam_min
        nu_min = ccc / lam_max
        ax.axvspan(nu_min, nu_max, color=colors[band], alpha=0.3, zorder=0)
        xmid = 10 ** ((np.log10(nu_min) + np.log10(nu_max)) / 2)
        ax.text(xmid, 1.02, band, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom")

    # Data + envelopes + best fits
    plt.loglog(nus, nuLnus, color="black", lw=2, label=r"$L_{\nu,\mathrm{avg}}$")

    plt.fill_between(
        nus, bb_ouv_min, bb_ouv_max,
        color="blue", alpha=0.25, label=r"OUV allowed range"
    )
    plt.loglog(nus, bb_ouv, color="blue", lw=2.5, label=r"$L_{\mathrm{bb,optUV}}$")

    if has_xray:
        plt.fill_between(
            nus, bb_x_min, bb_x_max,
            color="red", alpha=0.25, label=r"Soft X allowed range"
        )
        plt.loglog(nus, bb_x, color="red", lw=2.5, label=r"$L_{\mathrm{bb,softX}}$")

    plt.xlabel(r"Frequency $\nu$ (Hz)")
    plt.ylabel(r"$\nu L_{\nu}\;(\mathrm{erg\,s^{-1}})$")
    plt.legend()
    plt.grid(True, which="both", ls="--", lw=0.5)

    ymax = np.max(bb_ouv_max)
    plt.ylim(ymax * 1e-10, ymax * 1e2)
    plt.xlim(nus.min(), nus.max())
    save_name = f"{spec_num}_chisquarecompare.png"
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_name}")