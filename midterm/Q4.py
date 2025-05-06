import numpy as np
import matplotlib.pyplot as plt
import glob

# ------------------------
# Parameters
# ------------------------
n = 256
dx = 1.0 / n
c = 0.2
dt = c * dx
t_final = 1.0
steps = int(t_final / dt)
k0 = 24
N, M = 3, 3
x = np.linspace(0, 1, n+1)
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
'''
# ------------------------
# Numerical Schemes
# ------------------------
def DRP(u):
    u_ext = np.concatenate([u[-3:], u, u[:3]])
    a = np.array([-0.02651995, 0.18941314, -0.79926643, 0.0,
                   0.79926643, -0.18941314, 0.02651995])
    f = np.zeros_like(u)
    for i in range(n+1):
        for j in range(-N, M+1):
            idx =(i+j)%n
            f[i] += a[j + N] * u[idx]
    return f / dx


def DRP_M(u):
    a = np.array([-0.020843142770, 0.1667052380518, -0.770882380518, 0.0,
                   0.770882380518, -0.1667052380518, 0.020843142770])
    c = np.array([0.014281184692, -0.086150669577, 0.235718815308,
                  0.327698660846, 0.235718815308, -0.086150669577, 0.014281184692])
    f = np.zeros_like(u)
    u_stencil = np.abs(np.max(u) - np.min(u))
    for i in range(n+1):
        for j in range(-N, M+1):
            idx = (i + j) % n
            f[i] += a[j + N] * u[idx]/dx + 1e-7 * n * n * u_stencil * c[j + N] * u[idx]
    return f

def MDCD(u):
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


def SA_DRP(u):
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

def save_result(x, u, scheme_name, t_final):
    filename = f"{scheme_name.lower().replace('-', '_')}_t{int(t_final)}_new.npz"
    np.savez(filename, x=x, u=u, scheme=scheme_name, t=t_final)
    print(f"Saved {scheme_name} result to {filename}")


def rk4_step(u, compute_dudx):
    k1 = -compute_dudx(u)
    k2 = -compute_dudx(u + 0.5 * dt * k1)
    k3 = -compute_dudx(u + 0.5 * dt * k2)
    k4 = -compute_dudx(u + dt * k3)
    return u + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)


schemes = {
    "DRP": DRP,
    "DRP_M": DRP_M,
    "MDCD": MDCD,
    "SA-DRP": SA_DRP,
}

results = {}
for name, scheme in schemes.items():
    u = initial_condition(x)
    for step in range(steps):
        u = rk4_step(u, scheme)
    results[name] = u.copy()
    save_result(x, u, scheme_name=name, t_final=t_final)

'''

def load_and_plot_results_colored():
    display_order = ["DRP", "DRP-M", "MDCD", "SA-DRP"]
    label_map = {name.lower().replace("-", "_"): name for name in display_order}
    marker_map = {
        "DRP": 's',       
        "DRP-M": '^',     
        "MDCD": 'o',      
        "SA-DRP": 'd'     
    }
    color_map = {
        "DRP": 'tab:blue',
        "DRP-M": 'tab:orange',
        "MDCD": 'tab:green',
        "SA-DRP": 'tab:red'
    }

    files = sorted(glob.glob("*_t1_new.npz"))
    data_dict = {}
    errors = {}

    for file in files:
        data = np.load(file)
        key = str(data["scheme"]).lower().replace("-", "_")
        if key in label_map:
            data_dict[key] = {
                "x": data["x"],
                "u": data["u"],
                "scheme": label_map[key],
                "t": float(data["t"])
            }

    if not data_dict:
        print("No matching .npz results found.")
        return

    plt.figure(figsize=(10, 6))
    plt.xlim([0.0, 1.0])
    t_final = list(data_dict.values())[0]["t"]
    x = list(data_dict.values())[0]["x"]
    x_exact = (x - t_final) % 1.0
    u_exact = initial_condition(x_exact)

    for scheme_name in display_order:
        file_key = scheme_name.lower().replace("-", "_")
        if file_key in data_dict:
            u = data_dict[file_key]["u"]
            plt.plot(x[:], u[:],
                     marker=marker_map[scheme_name],
                     linestyle='None',
                     color=color_map[scheme_name],
                     markerfacecolor='white',  # 空心效果
                     markeredgecolor=color_map[scheme_name],
                     label=scheme_name,
                     markersize=5,
                     linewidth=1)
            # Compute L1 error
            errors[scheme_name] = np.mean(np.abs(u - u_exact))

    plt.plot(x[:], u_exact[:], 'k-', linewidth=1.5, label="Exact")

    plt.xlabel("x")
    plt.ylabel("u")
    
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("test.png")
    plt.show()

    # Print errors after plot
    print("\nL1 Errors at t=1.0:")
    for name in display_order:
        if name in errors:
            print(f"{name:<8}: {errors[name]:.6e}")
load_and_plot_results_colored()