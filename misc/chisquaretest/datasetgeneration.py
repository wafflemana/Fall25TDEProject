import numpy as np

# Constants
hhh = 6.626 * 10**(-34) #plank's constant Js
ccc = 3 * 10**8         #speed of light m/s
kbb = 1.38 * 10**(-23)  #boltzmann constant J/K
ttt = 300              #temperature in Kelvin


# Frequency range: 1e11 to 1.5e12 Hz (250 points)
nus = np.linspace(1e12, 1e14, 250)

# Plank's law
def planck(nu, T):
    return (2 * hhh * nu**3) / (ccc**2) / (np.exp(hhh * nu / (kbb * ttt)) - 1)

I_bb = planck(nus, ttt)

# adding noise to data and cleaning
noise = np.random.normal(loc=0.0, scale=0.03 * I_bb)
I_obs = I_bb + noise
I_obs[I_obs < 0] = 0

# file saved
with open("blackbody_sample_large.dat", "w") as f:
    for nu, I in zip(nus, I_obs):
        f.write(f"{nu:.6e}  0  {I:.6e}\n")