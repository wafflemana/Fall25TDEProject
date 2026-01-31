import matplotlib.pyplot as plt
import pandas as pd
import os
import glob

#constants
ccc = 3.00 * 10**10      # speed of light [cm/s]
sbc = 1.38 * 10**(-16)   # k_B (Boltzmann constant) [erg/K]
hhh = 6.625 * 10**(-27)  # Planck's constant [erg*s]
CCC = 5.879 * 10**10     #Wein displacement law proportionality constant [Hz/K]
BHmass = 10**7 * (1.989 * 10**33) #Black hole mass [g]
GGG =  6.67 * 10**(-8)   #Gravitational constant [cm^3/g s^2]
rrr_g = (GGG * BHmass) / ccc**2 #Gravitational radius of the black hole


#LaTex rendering
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

#load csv
csv_path = r"C:\Users\aviye\PycharmProjects\Fall25TDEProject\models\m7am0.9_M22\chisquareresults\chisquare_results_m7am0.9_M22.csv"
save_dir = r"C:\Users\aviye\PycharmProjects\Fall25TDEProject\models\m7am0.9_M22\chisquarecomparison"
data = pd.read_csv(csv_path)

#Column data from csv
time = data["Spec_File"]
r_optuv = data["OptUV_Radius_cm"]
T_optuv = data["OptUV_Temperature_K"]
r_xray = data["SoftX_Radius_cm"]
T_xray = data["SoftX_Temperature_K"]

#Convert radius to units of r_g
r_optuv_rg = r_optuv / rrr_g
r_xray_rg = r_xray / rrr_g

#2 panel figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Temperature vs time
ax1.plot(time, T_optuv, label=r"Optical/UV", color="blue", lw=2)
ax1.plot(time, T_xray, label=r"Soft X-ray", color="red", lw=2)
ax1.set_ylabel(r"$T_{\mathrm{bb}}$ (K)")
ax1.set_title(r"Best Fit Parameters vs Time")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Radius vs time (in r_g)
ax2.plot(time, r_optuv_rg, label=r"Optical/UV", color="blue", lw=2)
ax2.plot(time, r_xray_rg, label=r"Soft X-ray", color="red", lw=2)
ax2.set_xlabel(r"$t$")
ax2.set_ylabel(r"$R_{\mathrm{bb}}$ ($r_g$)")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Remove numeric tick labels on x-axis
ax2.set_xticks([])

#save fig and show it
save_path = os.path.join(save_dir, f"bestfitparamsvstime_m7am0.9_M22.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()


