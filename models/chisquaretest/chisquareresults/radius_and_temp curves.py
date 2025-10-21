import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter

# === Enable LaTeX rendering ===
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12
})

# === Load CSV data ===
csv_path = r"C:\Users\Aviyel\PycharmProjects\Fall25TDEProject\models\chisquaretest\chisquareresults\chisquare_results.csv"
data = pd.read_csv(csv_path)

# === Extract columns ===
time = data["Spec_File"]
r_optuv = data["OptUV_Radius"]
T_optuv = data["OptUV_Temperature"]
r_xray = data["SoftX_Radius"]
T_xray = data["SoftX_Temperature"]

# === Create figure with 2 panels ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# temperature vs t curves
ax1.plot(time, T_optuv, label=r"Optical/UV", color="blue", lw=2)
ax1.plot(time, T_xray, label=r"Soft X-ray", color="red", lw=2)
ax1.set_ylabel(r"$T_{\mathrm{bb}}$ (K)")
ax1.set_title(r"[Title Pending ASF]")
ax1.legend()
ax1.grid(True, alpha=0.3)

# radius vs t curves
ax2.plot(time, r_optuv, label=r"Optical/UV", color="blue", lw=2)
ax2.plot(time, r_xray, label=r"Soft X-ray", color="red", lw=2)
ax2.set_xlabel(r"$t$")
ax2.set_ylabel(r"$R_{\mathrm{bb}}$ (cm)")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Remove numeric tick labels on x-axis
ax2.set_xticks([])

plt.tight_layout()
plt.show()
