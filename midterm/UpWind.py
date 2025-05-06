import numpy as np
import matplotlib.pyplot as plt
import glob
import re
# ------------------------
# Parameters
# ------------------------
c = 0.2
t_final = 1.0

k0 = 24
N, M = 3, 3

m=64
# Random phase and energy spectrum
np.random.seed(42)
psi_k = np.random.rand(m)
def E_k(k): 
    return (k/k0)**4*np.exp(-2 * (k/k0)**2)

def initial_condition(x):
    u = np.ones_like(x)
    for k in range(1, m + 1):
        u += 0.1 * np.sqrt(E_k(k)) * np.sin(2 * np.pi * k * (x + psi_k[k-1]))
    return u

# ------------------------
# Numerical Schemes
# ------------------------
def DRP(u,dx):
    n = len(u)-1
    u_ext = np.concatenate([u[-3:], u, u[:3]])
    a = np.array([-0.02651995, 0.18941314, -0.79926643, 0.0,
                   0.79926643, -0.18941314, 0.02651995])
    f = np.zeros_like(u)
    for i in range(n+1):
        for j in range(-3, 3+1):
            idx =(i+j)%n
            f[i] += a[j + 3] * u[idx]
    return f / dx


def DRP_M(u,dx):
    n = len(u)-1
    a = np.array([-0.020843142770, 0.1667052380518, -0.770882380518, 0.0,
                   0.770882380518, -0.1667052380518, 0.020843142770])
    c = np.array([0.014281184692, -0.086150669577, 0.235718815308,
                  0.327698660846, 0.235718815308, -0.086150669577, 0.014281184692])
    f = np.zeros_like(u)
    u_stencil = np.abs(np.max(u) - np.min(u))
    for i in range(n+1):
        for j in range(-N, M+1):
            idx = (i + j) % n
            f[i] += a[j + N] * u[idx]/dx + dx * u_stencil * c[j + N] * u[idx]
    return f

def MDCD(u,dx):
    n = len(u)-1
    f = np.zeros_like(u)
    disp = 0.0463783
    diss = 0.001
    u_stencil = np.abs(np.max(u) - np.min(u))
    a = np.array([-1/2*disp-1/2*diss,2*disp+3*diss+1/12,-5/2*disp-15/2*diss-2/3,
                  10*diss,
                  5/2*disp-15/2*diss+2/3,-2*disp+3*diss-1/12,1/2*disp-1/2*diss])
    for i in range(n+1):
        for j in range(-N, M+1):
            idx = (i + j) % n
            f[i] += a[j + N] * u[idx]
    return f / dx

def upwind_3rd(u, dx):
    dudx = np.zeros_like(u)
    # 周期性边界拓展：2个点
    u_ext = np.concatenate([u[-2:], u, u[:2]])
    for j in range(len(u)):
        dudx[j] = (2*u_ext[j+3] + 3*u_ext[j+2] - 6*u_ext[j+1] + u_ext[j]) / (6*dx)
    return 1.0 * dudx

def upwind_2rd(u, dx):
    dudx = np.zeros_like(u)
    # 周期性边界拓展：2个点
    u_ext = np.concatenate([u[-1:], u, u[:1]])
    for j in range(len(u)):
        dudx[j] = (3*u_ext[j+1] - 4*u_ext[j] + u_ext[j-1]) / (2*dx)
    return 1.0 * dudx

def upwind_1rd(u, dx):
    dudx = np.zeros_like(u)
    # 周期性边界拓展：2个点
    u_ext = np.concatenate([u[-1:], u, u[:1]])
    for j in range(len(u)):
        dudx[j] = (u_ext[j+1] -  u_ext[j]) / (dx)
    return dudx

def _wrap(arr, shift):
    new = np.zeros_like(arr)
    for i in range(len(arr)):
        idx = (i + shift) % (len(arr)-1)
        new[i] = arr[idx]
    return new


def _effective_k(u, dx):
    e = 1e-8
    S1 = _wrap(u, +1) - 2*u + _wrap(u, -1)
    S2 = (_wrap(u, +2) - 2*u + _wrap(u, -2)) / 4.0
    S3 = _wrap(u, +2) - 2*_wrap(u, +1) + u
    S4 = (_wrap(u, +3) - 2*_wrap(u, +1) + _wrap(u, -1)) / 4.0
    C1 = _wrap(u, +1) - u
    C2 = (_wrap(u, +2) - _wrap(u, -1)) / 3.0
    num = np.abs(np.abs(S1+S2) - np.abs(S1-S2))+np.abs(np.abs(S3+S4) - np.abs(S3-S4))+np.abs(np.abs(C1+C2) - np.abs(C1-C2)/2.0)+ 2*e
    den = (np.abs(S1+S2) + np.abs(S1-S2) +
           np.abs(S3+S4) + np.abs(S3-S4) +
           np.abs(C1+C2) + np.abs(C1-C2) + e)
    ratio = np.clip(2*np.minimum(num/den, 1.0)-1.0, -1.0, 1.0)
    return np.arccos(ratio)

def _cdisp(k):
    c = np.empty_like(k)
    small = k < 0.01
    mid = (k >= 0.01) & (k < 2.5)
    big = k >= 2.5
    c[small] = 1/30.0
    ks = k[mid]
    c[mid] = (ks + (1/6)*np.sin(2*ks) - (4/3)*np.sin(ks)) / \
             (np.sin(3*ks) - 4*np.sin(2*ks) + 5*np.sin(ks))
    c[big] = 0.1985842
    return c

def _cdiss(k, a):
    c = np.where(k <= 1.0,
                 0.001,
                 np.minimum(0.012,0.001 + 0.011*np.sqrt(np.maximum(k-1.0, 0.0)/(np.pi-1))))
    c = np.minimum(c, 0.012)
    return np.sign(a) * c           


def SA_DRP(u,dx):
    n = len(u)-1
    k_esw = _effective_k(u, dx)
    disp = _cdisp(k_esw)
    diss = _cdiss(k_esw, a=1.0)
    f_right = (
        (1/2 * disp + 1/2 * diss)       * _wrap(u, -2) +
        (-3/2 * disp - 5/2 * diss - 1/12) * _wrap(u, -1) +
        (disp + 5 * diss + 7/12)          * u +
        (disp - 5 * diss + 7/12)          * _wrap(u, +1) +
        (-3/2 * disp + 5/2 * diss - 1/12) * _wrap(u, +2) +
        (1/2 * disp - 1/2 * diss)         * _wrap(u, +3)
    )
    dudx = f_right - _wrap(f_right, -1)
    return dudx / dx

def save_result(x, u, scheme_name, t_final,grid):
    filename = f"{scheme_name.lower().replace('-', '_')}_t{int(t_final)}_n{int(grid)}_.npz"
    np.savez(filename, x=x, u=u, scheme=scheme_name, t=t_final)
    print(f"Saved {scheme_name} result to {filename}")


def rk4_step(u, compute_dudx,dx,dt):
    k1 = -compute_dudx(u,dx)
    k2 = -compute_dudx(u + 0.5 * dt * k1,dx)
    k3 = -compute_dudx(u + 0.5 * dt * k2,dx)
    k4 = -compute_dudx(u + dt * k3,dx)
    return u + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

schemes = {
    "Up1":upwind_1rd,
}
grid_sizes = [4096]

for scheme_name, scheme_func in schemes.items():
    for n in grid_sizes:
        dx = 1.0 / n
        dt = c * dx
        steps = int(t_final / dt)
        x = np.linspace(0, 1, n+1)
        u = initial_condition(x)

        for i in range(steps):
            u = rk4_step(u, scheme_func, dx, dt)
            if i%100==0:
                print(dt*i)

        x_exact = (x - t_final) % 1.0
        u_exact = initial_condition(x_exact)
        l1_error = np.mean(np.abs(u - u_exact))

        # Print error
        print(f"{scheme_name:>7} | N={n:<4} | L1 Error = {l1_error:.4e}")
        
        # Plot
        plt.figure(figsize=(6, 4))
        plt.plot(x, u, 'b-', label=f"{scheme_name} (N={n})")
        plt.plot(x, u_exact, 'k--', label="Exact")
        plt.xlabel("x")
        plt.ylabel("u")
        plt.title(f"{scheme_name} | Grid={n} | Error={l1_error:.2e}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{scheme_name.lower().replace('-', '_')}_n{n}_result.png")
        plt.close()
        