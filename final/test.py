import numpy as np
import matplotlib.pyplot as plt

# Grid resolution
Mx = 128
My = 128
delta_eta = 1.0 / Mx
delta_xi = 1.0 / My

# Define airfoil coordinates (NACA 4-digit symmetric profile)
def naca_airfoil(x, thickness=0.12):
    """Generate NACA symmetric airfoil coordinates"""
    #t = thickness
    #y = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    y = 0.594689181 * (0.298222773 * np.sqrt(x) - 0.127125232 * x - 0.357907906 * x ** 2 + 0.291984971 * x**3 - 0.105174606 * x**4)
    return y

# Create airfoil boundary (inner boundary)
x_airfoil = np.linspace(0, 1, Mx + 1)
y_upper = naca_airfoil(x_airfoil)
y_lower = -naca_airfoil(x_airfoil)

# Create full airfoil boundary (upper surface + lower surface in reverse)
airfoil_x = np.concatenate([x_airfoil, x_airfoil[::-1]])
airfoil_y = np.concatenate([y_upper, y_lower[::-1]])
print(airfoil_x)
input()
# Ensure we have exactly Mx+1 points by taking every other point if needed
if len(airfoil_x) > Mx + 1:
    indices = np.linspace(0, len(airfoil_x)-1, Mx + 1, dtype=int)
    airfoil_x = airfoil_x[indices]
    airfoil_y = airfoil_y[indices]

# Define outer boundary (rectangular far-field)
L = 20.0  # Distance to far-field
outer_boundary = np.zeros((Mx + 1, 2))

# Create outer boundary points corresponding to airfoil points
for i in range(Mx + 1):
    # Map airfoil point to far-field
    theta = np.arctan2(airfoil_y[i], airfoil_x[i] - 0.5) if airfoil_x[i] != 0.5 else (np.pi/2 if airfoil_y[i] > 0 else -np.pi/2)
    outer_boundary[i, 0] = 0.5 + L * np.cos(theta)
    outer_boundary[i, 1] = L * np.sin(theta)

# Initialize physical grid
physical_plane = np.zeros((Mx, My + 1, 2))

# Set boundary conditions
# Inner boundary (j=0): airfoil
for i in range(Mx):
    physical_plane[i, 0, 0] = airfoil_x[i]
    physical_plane[i, 0, 1] = airfoil_y[i]

# Outer boundary (j=My): far-field
for i in range(Mx):
    physical_plane[i, My, 0] = outer_boundary[i, 0]
    physical_plane[i, My, 1] = outer_boundary[i, 1]

# Initialize interior points with linear interpolation
for i in range(Mx):
    for j in range(1, My):
        t = j / My
        physical_plane[i, j, 0] = (1-t) * physical_plane[i, 0, 0] + t * physical_plane[i, My, 0]
        physical_plane[i, j, 1] = (1-t) * physical_plane[i, 0, 1] + t * physical_plane[i, My, 1]

# Jacobi iteration for grid smoothing
max_iter = 5000
tolerance = 1e-8
new_plane = physical_plane.copy()

# Create mask for boundary points (fixed)
fixed_mask = np.zeros((Mx, My + 1), dtype=bool)
fixed_mask[:, 0] = True    # Inner boundary (airfoil)
fixed_mask[:, My] = True   # Outer boundary (far-field)

print("Starting Jacobi iteration...")

for iteration in range(max_iter):
    max_error = 0.0
    
    for i in range(Mx):  # Skip boundary points in xi direction
        for j in range(1, My):  # Skip boundary points in eta direction
            
            if fixed_mask[i, j]:
                continue
            
            # Calculate grid metrics using central differences
            # Handle periodic boundary conditions in xi direction
            ip1 = (i + 1) % (Mx) #if i == Mx - 1 else i + 1
            im1 = (i - 1) % (Mx) #if i == 0 else i - 1
            
            # Derivatives with respect to xi (circumferential direction)
            d_xi_x = (physical_plane[ip1, j, 0] - physical_plane[im1, j, 0]) / (2 * delta_xi)
            d_xi_y = (physical_plane[ip1, j, 1] - physical_plane[im1, j, 1]) / (2 * delta_xi)
            
            # Derivatives with respect to eta (radial direction)
            d_eta_x = (physical_plane[i, j + 1, 0] - physical_plane[i, j - 1, 0]) / (2 * delta_eta)
            d_eta_y = (physical_plane[i, j + 1, 1] - physical_plane[i, j - 1, 1]) / (2 * delta_eta)
            
            # Grid metrics
            alpha = d_eta_x**2 + d_eta_y**2
            beta = d_xi_x * d_eta_x + d_xi_y * d_eta_y
            gamma = d_xi_x**2 + d_xi_y**2
            
            # Coefficients for the elliptic grid generation equations
            a11 = alpha / (delta_xi**2)
            a22 = gamma / (delta_eta**2)
            a12 = -beta / (2 * delta_xi * delta_eta)
            
            # Cross-derivative terms
            cross_term_x = a12 * (physical_plane[ip1, j + 1, 0] - physical_plane[ip1, j - 1, 0] - 
                                 physical_plane[im1, j + 1, 0] + physical_plane[im1, j - 1, 0])
            cross_term_y = a12 * (physical_plane[ip1, j + 1, 1] - physical_plane[ip1, j - 1, 1] - 
                                 physical_plane[im1, j + 1, 1] + physical_plane[im1, j - 1, 1])
            
            # New coordinates using the elliptic grid generation equations
            denom = 2 * (a11 + a22)
            
            x_new = (a11 * (physical_plane[ip1, j, 0] + physical_plane[im1, j, 0]) +
                    a22 * (physical_plane[i, j + 1, 0] + physical_plane[i, j - 1, 0]) +
                    cross_term_x) / denom
                    
            y_new = (a11 * (physical_plane[ip1, j, 1] + physical_plane[im1, j, 1]) +
                    a22 * (physical_plane[i, j + 1, 1] + physical_plane[i, j - 1, 1]) +
                    cross_term_y) / denom
            
            # Calculate error
            error_x = abs(x_new - physical_plane[i, j, 0])
            error_y = abs(y_new - physical_plane[i, j, 1])
            error = max(error_x, error_y)
            max_error = max(max_error, error)
            
            # Update
            new_plane[i, j, 0] = x_new
            new_plane[i, j, 1] = y_new
    
    # Apply updates
    physical_plane = new_plane.copy()
    
    # Print progress every 500 iterations
    if (iteration + 1) % 500 == 0:
        print(f"Iteration {iteration + 1}: max_error = {max_error:.2e}")
    
    # Check convergence
    if max_error < tolerance:
        print(f"Converged in {iteration + 1} iterations with error {max_error:.2e}")
        break
else:
    print(f"Did not converge after {max_iter} iterations. Final error: {max_error:.2e}")
'''
# Plotting
plt.figure(figsize=(12, 8))

# Plot grid lines
for j in range(0, My + 1, 2):  # Plot every other eta line
    plt.plot(physical_plane[:, j, 0], physical_plane[:, j, 1], 'b-', alpha=0.6, linewidth=0.5)

for i in range(0, Mx, 2):  # Plot every other xi line
    plt.plot(physical_plane[i, :, 0], physical_plane[i, :, 1], 'r-', alpha=0.6, linewidth=0.5)

# Highlight boundaries
plt.plot(physical_plane[:, 0, 0], physical_plane[:, 0, 1], 'k-', linewidth=2, label='Airfoil')
plt.plot(physical_plane[:, My, 0], physical_plane[:, My, 1], 'g-', linewidth=1, label='Far-field')

plt.title('Corrected Structured Grid Around Airfoil')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(-2, 2)
plt.ylim(-2, 2)

plt.tight_layout()
plt.savefig("jacobi_grid.png")
plt.show()

# Print some grid quality metrics
print(f"\nGrid dimensions: {Mx+1} x {My+1}")
print(f"Airfoil chord length: {np.max(airfoil_x) - np.min(airfoil_x):.3f}")
print(f"Max airfoil thickness: {np.max(y_upper) - np.min(y_lower):.3f}")
'''
# ===== 保存数据部分 =====
import numpy as np
import matplotlib.pyplot as plt

# 在你的主程序最后添加保存代码
def save_grid_data(physical_plane, Mx, My, filename='grid_data'):
    """保存网格数据"""
    
    # 方法1: 使用numpy的.npz格式（推荐）
    np.savez_compressed(f'{filename}.npz', 
                       physical_plane=physical_plane,
                       Mx=Mx, 
                       My=My)
    
    # 方法2: 使用numpy的.npy格式
    np.save(f'{filename}_array.npy', physical_plane)
    
    # 方法3: 保存为文本文件（可读性好但文件较大）
    # 重新整形为2D数组便于保存
    reshaped_data = physical_plane.reshape(-1, 2)
    np.savetxt(f'{filename}.txt', reshaped_data, 
               header=f'Grid data: Mx={Mx}, My={My}, Shape: {physical_plane.shape}',
               fmt='%.8f')
    
    print(f"Grid data saved as:")
    print(f"- {filename}.npz (compressed, recommended)")
    print(f"- {filename}_array.npy (numpy array)")
    print(f"- {filename}.txt (text format)")

# 在你的代码最后调用保存函数
save_grid_data(physical_plane, Mx, My, 'airfoil_grid')

# ===== 读取和绘制数据部分 =====
def load_and_plot_grid(filename='grid_data'):
    """读取并绘制网格数据"""
    
    # 方法1: 从.npz文件读取（推荐）
    try:
        data = np.load(f'{filename}.npz')
        physical_plane = data['physical_plane']
        Mx = int(data['Mx'])
        My = int(data['My'])
        print(f"Loaded from {filename}.npz")
        print(f"Grid dimensions: {Mx} x {My}")
        print(f"Array shape: {physical_plane.shape}")
        
    except FileNotFoundError:
        # 方法2: 从.npy文件读取
        try:
            physical_plane = np.load(f'{filename}_array.npy')
            # 需要手动设置Mx, My
            Mx, My = 128, 128  # 根据你的原始设置
            print(f"Loaded from {filename}_array.npy")
            
        except FileNotFoundError:
            # 方法3: 从文本文件读取
            print(f"Loading from {filename}.txt...")
            data = np.loadtxt(f'{filename}.txt')
            # 需要手动重新整形
            Mx, My = 128, 128  # 根据你的原始设置
            physical_plane = data.reshape(Mx, My + 1, 2)
    
    # 绘制网格
    plt.figure(figsize=(12, 8))
    
    # 绘制eta方向的网格线（径向线）
    for j in range(0, My + 1, 2):
        # 添加首尾连接形成闭合曲线
        x_coords = np.append(physical_plane[:, j, 0], physical_plane[0, j, 0])
        y_coords = np.append(physical_plane[:, j, 1], physical_plane[0, j, 1])
        plt.plot(x_coords, y_coords, 'b-', alpha=0.6, linewidth=0.5)
    
    # 绘制xi方向的网格线（周向线）
    for i in range(0, Mx, 2):
        plt.plot(physical_plane[i, :, 0], physical_plane[i, :, 1], 'r-', alpha=0.6, linewidth=0.5)
    
    # 突出显示边界
    # 翼型边界（内边界）- 需要闭合
    airfoil_x = np.append(physical_plane[:, 0, 0], physical_plane[0, 0, 0])
    airfoil_y = np.append(physical_plane[:, 0, 1], physical_plane[0, 0, 1])
    plt.plot(airfoil_x, airfoil_y, 'k-', linewidth=2, label='Airfoil')
    
    # 远场边界（外边界）- 需要闭合
    farfield_x = np.append(physical_plane[:, My, 0], physical_plane[0, My, 0])
    farfield_y = np.append(physical_plane[:, My, 1], physical_plane[0, My, 1])
    plt.plot(farfield_x, farfield_y, 'g-', linewidth=1, label='Far-field')
    
    plt.title('Loaded Structured Grid Around Airfoil')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    
    plt.tight_layout()
    plt.savefig("loaded_grid.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return physical_plane, Mx, My

# 使用示例：读取并绘制
physical_plane_loaded, Mx_loaded, My_loaded = load_and_plot_grid('airfoil_grid')

# ===== 额外的数据分析函数 =====
def analyze_grid_quality(physical_plane, Mx, My):
    """分析网格质量"""
    
    # 计算网格间距
    min_spacing = float('inf')
    max_spacing = 0
    
    for i in range(Mx):
        for j in range(My):
            # 计算相邻点之间的距离
            if j < My:
                # eta方向
                dx = physical_plane[i, j+1, 0] - physical_plane[i, j, 0]
                dy = physical_plane[i, j+1, 1] - physical_plane[i, j, 1]
                spacing = np.sqrt(dx**2 + dy**2)
                min_spacing = min(min_spacing, spacing)
                max_spacing = max(max_spacing, spacing)
            
            if i < Mx - 1:
                # xi方向
                dx = physical_plane[i+1, j, 0] - physical_plane[i, j, 0]
                dy = physical_plane[i+1, j, 1] - physical_plane[i, j, 1]
                spacing = np.sqrt(dx**2 + dy**2)
                min_spacing = min(min_spacing, spacing)
                max_spacing = max(max_spacing, spacing)
    
    print(f"\nGrid Quality Analysis:")
    print(f"Minimum grid spacing: {min_spacing:.6f}")
    print(f"Maximum grid spacing: {max_spacing:.6f}")
    print(f"Spacing ratio (max/min): {max_spacing/min_spacing:.2f}")

# 分析网格质量
analyze_grid_quality(physical_plane_loaded, Mx_loaded, My_loaded)