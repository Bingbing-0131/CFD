import numpy as np
import matplotlib.pyplot as plt

# List of files you saved from C++
#files = ["Lax_Wendroff/output_step_0.txt","Lax_Wendroff/output_step_20.txt", "Lax_Wendroff/output_step_200.txt", "Lax_Wendroff/output_step_2000.txt"]
#files = ["Warming_Beam/output_step_0.txt","Warming_Beam/output_step_20.txt", "Warming_Beam/output_step_200.txt", "Warming_Beam/output_step_2000.txt"]
files = ["Three_Order/output_step_0.txt","Three_Order/output_step_20.txt", "Three_Order/output_step_200.txt", "Three_Order/output_step_2000.txt"]

# Optional: add corresponding time labels for the legend
times = [0.0,0.1, 1.0,10.0]

# Plot setup
plt.figure(figsize=(8, 5))

for filename, t in zip(files, times):
    data = np.loadtxt(filename)
    x = data[:, 0] - 0.5
    u = data[:, 1]
    plt.plot(x, u, label=f"t = {t}")

plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title("Implicit Euler")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Three_Order.jpg")
