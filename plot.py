import numpy as np
import matplotlib.pyplot as plt

# List of files you saved from C++
#files = ["UpWind/output_step_0.txt","UpWind/output_step_20.txt", "UpWind/output_step_200.txt", "UpWind/output_step_2000.txt"]
files = ["Implicit_Euler/implicit_0.00.dat","Implicit_Euler/implicit_20.00.dat", "Implicit_Euler/implicit_200.00.dat","Implicit_Euler/implicit_2000.00.dat"]

# Optional: add corresponding time labels for the legend
times = [0.0,0.1, 1.0,10.0]

# Plot setup
plt.figure(figsize=(8, 5))

for filename, t in zip(files, times):
    data = np.loadtxt(filename)
    x = data[:, 0]
    u = data[:, 1]
    plt.plot(x, u, label=f"t = {t}")

plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title("Upwind Scheme: Periodic Boundary Conditions")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Implicit_Euler.jpg")
