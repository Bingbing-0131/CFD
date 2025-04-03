import numpy as np
import os
import matplotlib.pyplot as plt

class WarmingBeam:
    def __init__(self, mx, CFL, a, folder):
        self.mx = mx
        self.dx = 1.0 / mx
        self.CFL = CFL
        self.a = a
        self.dt = CFL * self.dx / a
        self.folder = folder

        self.u_prev = np.zeros(mx + 1)
        self.u_new = np.zeros(mx + 1)

        for i in range(mx + 1):
            x = i * self.dx
            self.u_prev[i] = self.initial_condition(x)
            self.u_new[i] = self.u_prev[i]

        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        sim.save_data(0)

    def initial_condition(self, x):
        x -= 0.5  # Center the domain
        return 1.0 if -0.25 <= x <= 0.25 else 0.0

    def forward(self):
        u = self.u_prev.copy()
        c = self.CFL

        u_m1 = np.roll(u, 1)   # circshift(u, 1)
        u_m2 = np.roll(u, 2)   # circshift(u, 2)

        self.u_new = u - 0.5 * c * (3*u - 4*u_m1 + u_m2) \
                    + 0.5 * c**2 * (u - 2*u_m1 + u_m2)

        self.u_prev = self.u_new.copy()


    def save_data(self, t):
        filename = os.path.join(self.folder, f"t_{t:.1f}.txt")
        with open(filename, 'w') as f:
            for i in range(self.mx + 1):
                f.write(f"{i * self.dx:.10e} {self.u_new[i]:.10e}\n")

    def visualize(self):
        plt.figure(figsize=(10, 6))
        colors = ['r', 'g', 'b']
        save_times = [0.1, 1.0, 10.0]

        for i, t in enumerate(save_times):
            filename = os.path.join(self.folder, f"t_{t:.1f}.txt")
            if not os.path.exists(filename):
                print(f"File not found: {filename}")
                continue

            data = np.loadtxt(filename)
            x, u = data[:, 0], data[:, 1]
            plt.plot(x, u, label=f't = {t}', color=colors[i % len(colors)])

        plt.xlabel('x')
        plt.ylabel('u')
        plt.title('Warming-Beam Scheme Results')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    sim = WarmingBeam(mx=100, CFL=0.5, a=1.0, folder="output")

    t = 0.0
    save_times = [0.1, 1.0, 10.0]
    save_index = 0
    step = 0
    while t <= save_times[-1] + 1e-6:
        if save_index < len(save_times) and abs(t - save_times[save_index]) < 1e-8:
            print(f"Saving data at t = {t:.1f}")
            sim.save_data(t)
            save_index += 1

        sim.forward()
        t += sim.dt
        step += 1

    sim.visualize()
