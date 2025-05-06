import numpy as np
import matplotlib.pyplot as plt

# 参数设定
alpha_5 = 0.01
alpha_1 = 0.071
alpha_2 = 0.054
alpha_3 = 0.046
alpha_4 = 0.042

'''
beta_1 = 0.0
beta_2 = 0.01
beta_3 = 0.05
beta_4 = 0.1
'''
# 波数范围
k = np.linspace(0, np.pi, 500)

# Re(k') 表达式
Re_k_prime_1 = ((4/3 + 5*alpha_1) * np.sin(k) + (-1/6 - 4*alpha_1) * np.sin(2*k) + alpha_1 * np.sin(3*k))/k-1
Re_k_prime_2 = ((4/3 + 5*alpha_2) * np.sin(k) + (-1/6 - 4*alpha_2) * np.sin(2*k) + alpha_2 * np.sin(3*k))/k-1
Re_k_prime_3 = ((4/3 + 5*alpha_3) * np.sin(k) + (-1/6 - 4*alpha_3) * np.sin(2*k) + alpha_3 * np.sin(3*k))/k-1
Re_k_prime_4 = ((4/3 + 5*alpha_4) * np.sin(k) + (-1/6 - 4*alpha_4) * np.sin(2*k) + alpha_4 * np.sin(3*k))/k-1
Re_k_prime_5 = ((4/3 + 5*alpha_5) * np.sin(k) + (-1/6 - 4*alpha_5) * np.sin(2*k) + alpha_5 * np.sin(3*k))/k-1

'''
# Im(k') 表达式，g(cos(k)) 可以替换为你实际的函数形式
g_cos_k = 4*np.cos(k)**3 -12*np.cos(k)**2 +12*np.cos(k)-4 # 示例函数，可修改
Im_k_prime_1 = beta_1 * g_cos_k
Im_k_prime_2 = beta_2 * g_cos_k
Im_k_prime_3 = beta_3 * g_cos_k
Im_k_prime_4 = beta_4 * g_cos_k
# 绘图
'''
plt.figure(figsize=(8, 5))

plt.plot(k, Re_k_prime_5, label=f"$alpha$={alpha_5}", color='green')
plt.plot(k, Re_k_prime_1, label=f"$alpha$={alpha_1}", color='blue')
plt.plot(k, Re_k_prime_2, label=f"$alpha$={alpha_2}", color='red')
plt.plot(k, Re_k_prime_3, label=f"$alpha$={alpha_3}", color='purple')
plt.plot(k, Re_k_prime_4, label=f"$alpha$={alpha_4}", color='orange')


'''
plt.plot(k, Im_k_prime_1, label=f"$alpha$={beta_1}", color='blue')
plt.plot(k, Im_k_prime_2, label=f"$alpha$={beta_2}", color='red')
plt.plot(k, Im_k_prime_3, label=f"$alpha$={beta_3}", color='purple')
plt.plot(k, Im_k_prime_4, label=f"$alpha$={beta_4}", color='orange')
'''
#plt.plot(k, Im_k_prime, label=r"$\mathrm{Im}(k')$", color='red', linestyle='--')
plt.xlabel(r"$k$")
plt.ylabel(r"$Re(k')/k - 1$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("real.png")
plt.show()

