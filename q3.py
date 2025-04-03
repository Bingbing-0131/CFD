import numpy as np
import matplotlib.pyplot as plt

# Define the x domain
x = np.linspace(0, np.pi, 1000)

# Define the functions
f0 = x
f1 = np.sin(x)
f2 = 0.5 * (4 * np.sin(x) - np.sin(2 * x))
f3 = (1/6) * (8 * np.sin(x) - np.sin(2 * x))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, f0, label='Exact', linewidth=2)
plt.plot(x, f1, label='First_order', linewidth=2)
plt.plot(x, f2, label='Second_order', linestyle='--')
plt.plot(x, f3, label='Third_order', linestyle=':')

# Style
plt.title('Re(k\')')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("real")


x = np.linspace(0, np.pi, 1000)

# Define the functions
f0 = 0*x
f1 = np.cos(x) - 1
f2 = 0.5 * (4 * np.cos(x) - 3 - np.cos(2 * x))
f3 = (1/6) * (4 * np.cos(x) - 3 - np.cos(2 * x))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, f0, label='Exact', linewidth=2)
plt.plot(x, f1, label='First_order', linewidth=2)
plt.plot(x, f2, label='Second_order', linestyle='--')
plt.plot(x, f3, label='Third_order', linestyle=':')

# Style
plt.title('Im(k\')')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("image")
