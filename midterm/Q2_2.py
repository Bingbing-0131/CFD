import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

# 波数范围
k = np.linspace(0, np.pi, 1000)

# 函数定义
def Re_k_prime(k, alpha):
    return (4/3 + 5*alpha) * np.sin(k) + (-1/6 - 4*alpha) * np.sin(2*k) + alpha * np.sin(3*k)

def integrand(k, alpha, nu):
    re_kp = Re_k_prime(k, alpha)
    return np.exp(nu * (np.pi - k)) * (re_kp - k)**2

# 参数范围
alpha_vals = np.linspace(-0.1, 0.1, 200)
nu_vals = [ 6,7,8,9,10]  # 你可以添加更多值

# 画图
plt.figure(figsize=(8, 5))

for nu in nu_vals:
    E_vals = []
    for alpha in alpha_vals:
        f = integrand(k, alpha, nu)
        E = simpson(f, k) / np.exp(nu * np.pi)
        E_vals.append(E)

    E_vals = np.array(E_vals)
    min_idx = np.argmin(E_vals)
    alpha_opt = alpha_vals[min_idx]
    E_min = E_vals[min_idx]

    plt.plot(alpha_vals, np.log10(E_vals), label=fr"$\nu={nu}$, min at $\alpha={alpha_opt:.4f}$")
    plt.axvline(alpha_opt, linestyle='--', linewidth=0.8, alpha=0.5)

plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\log_{10} E(\alpha)$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("alpha.png")
plt.show()
