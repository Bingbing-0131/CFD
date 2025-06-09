import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# Grid resolution
Mx = 128
My = 128
delta_eta = 1.0 / Mx
delta_xi = 1.0 / My
delta = 0.05

# Define airfoil coordinates (NACA 4-digit symmetric profile)
def naca_airfoil(x, thickness=0.12):
    """Generate NACA symmetric airfoil coordinates"""
    y = 0.594689181 * (0.298222773 * np.sqrt(x) - 0.127125232 * x - 0.357907906 * x ** 2 + 0.291984971 * x**3 - 0.105174606 * x**4)
    return y

def create_initial_grid():
    """创建初始网格"""
    print("Creating initial airfoil grid...")
    
    # Create airfoil boundary (inner boundary)
    x_airfoil = np.linspace(0, 1, Mx + 1)
    y_upper = naca_airfoil(x_airfoil)
    y_lower = -naca_airfoil(x_airfoil)

    # Create full airfoil boundary (upper surface + lower surface in reverse)
    airfoil_x = np.concatenate([x_airfoil, x_airfoil[::-1]])
    airfoil_y = np.concatenate([y_upper, y_lower[::-1]])

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
    physical_plane = np.zeros((Mx + 1, My + 1, 2))

    # Set boundary conditions
    # Inner boundary (j=0): airfoil
    for i in range(Mx + 1):
        physical_plane[i, 0, 0] = airfoil_x[i]
        physical_plane[i, 0, 1] = airfoil_y[i]

    # Outer boundary (j=My): far-field
    for i in range(Mx + 1):
        physical_plane[i, My, 0] = outer_boundary[i, 0]
        physical_plane[i, My, 1] = outer_boundary[i, 1]

    # Linear interpolation for initial guess
    for i in range(Mx + 1):
        for j in range(1, My):
            t = j / My
            physical_plane[i, j, 0] = (1-t) * physical_plane[i, 0, 0] + t * physical_plane[i, My, 0]
            physical_plane[i, j, 1] = (1-t) * physical_plane[i, 0, 1] + t * physical_plane[i, My, 1]

    return physical_plane

def smooth_initial_grid(physical_plane, plot_iterations=False):
    """第一阶段：使用标准椭圆方程平滑初始网格"""
    print("=== STAGE 1: Smoothing initial grid with standard elliptic equations ===")
    
    max_iter = 1000
    tolerance = 1e-8
    new_plane = physical_plane.copy()

    # Create mask for boundary points (fixed)
    fixed_mask = np.zeros((Mx + 1, My + 1), dtype=bool)
    fixed_mask[:, 0] = True    # Inner boundary (airfoil)
    fixed_mask[:, My] = True   # Outer boundary (far-field)

    # Create directory for stage 1 plots
    if plot_iterations:
        plot_dir = "stage1_plots"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
            print(f"Created directory: {plot_dir}")

    for iteration in range(max_iter):
        max_error = 0.0
        
        for i in range(1,Mx):  # Skip boundary points in xi direction
            for j in range(1, My):  # Skip boundary points in eta direction
                
                if fixed_mask[i, j]:
                    continue
                
                # Handle periodic boundary conditions in xi direction
                ip1 = (i + 1) % (Mx+1)
                im1 = (i - 1) % (Mx+1)
                
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
            print(f"Stage 1 - Iteration {iteration + 1}: max_error = {max_error:.2e}")
        
        # Check convergence
        if max_error < tolerance:
            print(f"Stage 1 converged in {iteration + 1} iterations with error {max_error:.2e}")
            break
    
    return physical_plane

def calculate_pq_coefficients(physical_plane):
    """计算P、Q控制函数 - 纯原始代码"""
    print("Calculating P,Q control functions...")
    
    P = np.zeros((Mx + 1, My + 1))
    Q = np.zeros((Mx + 1, My + 1))
    
    # 左下角 (0, 0)
    x_eta = (physical_plane[0, 1, 0] - physical_plane[0, 0, 0]) / delta_eta
    y_eta = (physical_plane[0, 1, 1] - physical_plane[0, 0, 1]) / delta_eta
    x_xi = (physical_plane[1, 0, 0] - physical_plane[0, 0, 0]) / delta_xi
    y_xi = (physical_plane[1, 0, 1] - physical_plane[0, 0, 1]) / delta_xi

    x_eta_eta = (physical_plane[0, 0, 0] - 2 * physical_plane[0, 1, 0] + physical_plane[0, 2, 0]) / (delta_eta * delta_eta)
    y_eta_eta = (physical_plane[0, 0, 1] - 2 * physical_plane[0, 1, 1] + physical_plane[0, 2, 1]) / (delta_eta * delta_eta)
    x_xi_xi = (physical_plane[0, 0, 0] - 2 * physical_plane[1, 0, 0] + physical_plane[2, 0, 0]) / (delta_xi * delta_xi)
    y_xi_xi = (physical_plane[0, 0, 1] - 2 * physical_plane[1, 0, 1] + physical_plane[2, 0, 1]) / (delta_xi * delta_xi)

    alpha = x_eta ** 2 + y_eta ** 2
    gamma = x_xi ** 2 + y_xi ** 2

    P[0, 0] = -(x_xi * x_xi_xi + y_xi * y_xi_xi) / gamma - (x_xi * x_eta_eta + y_xi * y_eta_eta) / alpha
    Q[0, 0] = -(x_eta * x_eta_eta + y_eta * y_eta_eta) / alpha - (x_eta * x_xi_xi + y_eta * y_xi_xi) / gamma

    # 右下角 (Mx, 0)
    x_eta = (physical_plane[Mx, 1, 0] - physical_plane[Mx, 0, 0]) / delta_eta
    y_eta = (physical_plane[Mx, 1, 1] - physical_plane[Mx, 0, 1]) / delta_eta
    x_xi = (physical_plane[Mx, 0, 0] - physical_plane[Mx-1, 0, 0]) / delta_xi
    y_xi = (physical_plane[Mx, 0, 1] - physical_plane[Mx-1, 0, 1]) / delta_xi

    x_eta_eta = (physical_plane[Mx, 0, 0] - 2 * physical_plane[Mx, 1, 0] + physical_plane[Mx, 2, 0]) / (delta_eta * delta_eta)
    y_eta_eta = (physical_plane[Mx, 0, 1] - 2 * physical_plane[Mx, 1, 1] + physical_plane[Mx, 2, 1]) / (delta_eta * delta_eta)
    x_xi_xi = (physical_plane[Mx-2, 0, 0] - 2 * physical_plane[Mx-1, 0, 0] + physical_plane[Mx, 0, 0]) / (delta_xi * delta_xi)
    y_xi_xi = (physical_plane[Mx-2, 0, 1] - 2 * physical_plane[Mx-1, 0, 1] + physical_plane[Mx, 0, 1]) / (delta_xi * delta_xi)

    alpha = x_eta ** 2 + y_eta ** 2
    gamma = x_xi ** 2 + y_xi ** 2

    P[Mx, 0] = -(x_xi * x_xi_xi + y_xi * y_xi_xi) / gamma - (x_xi * x_eta_eta + y_xi * y_eta_eta) / alpha
    Q[Mx, 0] = -(x_eta * x_eta_eta + y_eta * y_eta_eta) / alpha - (x_eta * x_xi_xi + y_eta * y_xi_xi) / gamma
    
    # 左上角 (0, My)
    x_eta = (physical_plane[0, My, 0] - physical_plane[0, My-1, 0]) / delta_eta
    y_eta = (physical_plane[0, My, 1] - physical_plane[0, My-1, 1]) / delta_eta
    x_xi = (physical_plane[1, My, 0] - physical_plane[0, My, 0]) / delta_xi
    y_xi = (physical_plane[1, My, 1] - physical_plane[0, My, 1]) / delta_xi

    x_eta_eta = (physical_plane[0, My-2, 0] - 2 * physical_plane[0, My-1, 0] + physical_plane[0, My, 0]) / (delta_eta * delta_eta)
    y_eta_eta = (physical_plane[0, My-2, 1] - 2 * physical_plane[0, My-1, 1] + physical_plane[0, My, 1]) / (delta_eta * delta_eta)
    x_xi_xi = (physical_plane[0, My, 0] - 2 * physical_plane[1, My, 0] + physical_plane[2, My, 0]) / (delta_xi * delta_xi)
    y_xi_xi = (physical_plane[0, My, 1] - 2 * physical_plane[1, My, 1] + physical_plane[2, My, 1]) / (delta_xi * delta_xi)

    alpha = x_eta ** 2 + y_eta ** 2
    gamma = x_xi ** 2 + y_xi ** 2

    P[0, My] = -(x_xi * x_xi_xi + y_xi * y_xi_xi) / gamma - (x_xi * x_eta_eta + y_xi * y_eta_eta) / alpha
    Q[0, My] = -(x_eta * x_eta_eta + y_eta * y_eta_eta) / alpha - (x_eta * x_xi_xi + y_eta * y_xi_xi) / gamma

    # 右上角 (Mx, My)
    x_eta = (physical_plane[Mx, My, 0] - physical_plane[Mx, My-1, 0]) / delta_eta
    y_eta = (physical_plane[Mx, My, 1] - physical_plane[Mx, My-1, 1]) / delta_eta
    x_xi = (physical_plane[Mx, My, 0] - physical_plane[Mx-1, My, 0]) / delta_xi
    y_xi = (physical_plane[Mx, My, 1] - physical_plane[Mx-1, My, 1]) / delta_xi

    x_eta_eta = (physical_plane[Mx, My-2, 0] - 2 * physical_plane[Mx, My-1, 0] + physical_plane[Mx, My, 0]) / (delta_eta * delta_eta)
    y_eta_eta = (physical_plane[Mx, My-2, 1] - 2 * physical_plane[Mx, My-1, 1] + physical_plane[Mx, My, 1]) / (delta_eta * delta_eta)
    x_xi_xi = (physical_plane[Mx-2, My, 0] - 2 * physical_plane[Mx-1, My, 0] + physical_plane[Mx, My, 0]) / (delta_xi * delta_xi)
    y_xi_xi = (physical_plane[Mx-2, My, 1] - 2 * physical_plane[Mx-1, My, 1] + physical_plane[Mx, My, 1]) / (delta_xi * delta_xi)

    alpha = x_eta ** 2 + y_eta ** 2
    gamma = x_xi ** 2 + y_xi ** 2

    P[Mx, My] = -(x_xi * x_xi_xi + y_xi * y_xi_xi) / gamma - (x_xi * x_eta_eta + y_xi * y_eta_eta) / alpha
    Q[Mx, My] = -(x_eta * x_eta_eta + y_eta * y_eta_eta) / alpha - (x_eta * x_xi_xi + y_eta * y_xi_xi) / gamma
    '''
    # 左边界 (xi = 0)
    for j in range(1, My):
        # eta方向导数使用中心差分
        x_eta = (physical_plane[0, j+1, 0] - physical_plane[0, j-1, 0]) / (2 * delta_eta)
        y_eta = (physical_plane[0, j+1, 1] - physical_plane[0, j-1, 1]) / (2 * delta_eta)
        
        x_eta_eta = (physical_plane[0, j+1, 0] - 2 * physical_plane[0, j, 0] + physical_plane[0, j-1, 0]) / (delta_eta * delta_eta)
        y_eta_eta = (physical_plane[0, j+1, 1] - 2 * physical_plane[0, j, 1] + physical_plane[0, j-1, 1]) / (delta_eta * delta_eta)
        
        # 边界一阶导数 (单侧差分)
        x_xi_boundary = (physical_plane[1, j, 0] - physical_plane[0, j, 0]) / delta_xi
        y_xi_boundary = (physical_plane[1, j, 1] - physical_plane[0, j, 1]) / delta_xi
        
        # 计算度量系数
        alpha = x_eta * x_eta + y_eta * y_eta  # alpha
        gamma = x_xi_boundary * x_xi_boundary + y_xi_boundary * y_xi_boundary  # gamma_boundary
        
        # 正交投影计算 (基于公式 6.12)
        # x_xi = a(a·x_xi^0) = (y_eta·x_xi^0 - x_eta·y_xi^0)/g11 * (y_eta, -x_eta)
        dot_product = y_eta * x_xi_boundary - x_eta * y_xi_boundary
        x_xi_orthogonal = y_eta * dot_product / alpha
        y_xi_orthogonal = -x_eta * dot_product / alpha
        
        # Ghost点位置计算 (基于公式)
        # x_{-1,j} = x_{0,j} - (x_xi)_{0,j}
        ghost_x = physical_plane[0, j, 0] - x_xi_orthogonal * delta_xi
        ghost_y = physical_plane[0, j, 1] - y_xi_orthogonal * delta_xi
        
        # 二阶导数计算
        x_xi_xi = (physical_plane[1, j, 0] - 2 * physical_plane[0, j, 0] + ghost_x) / (delta_xi * delta_xi)
        y_xi_xi = (physical_plane[1, j, 1] - 2 * physical_plane[0, j, 1] + ghost_y) / (delta_xi * delta_xi)
        
        # 更新度量系数
        gamma_orthogonal = x_xi_orthogonal * x_xi_orthogonal + y_xi_orthogonal * y_xi_orthogonal
        
        # P、Q系数计算
        P[0, j] = -(x_xi_boundary * x_xi_xi + y_xi_boundary * y_xi_xi) / gamma_orthogonal - (x_xi_boundary * x_eta_eta + y_xi_boundary * y_eta_eta) / alpha
        Q[0, j] = -(x_eta * x_eta_eta + y_eta * y_eta_eta) / alpha - (x_eta * x_xi_xi + y_eta * y_xi_xi) / gamma_orthogonal

    # 右边界 (xi = Mx)
    for j in range(1, My):
        # eta方向导数使用中心差分
        x_eta = (physical_plane[Mx, j+1, 0] - physical_plane[Mx, j-1, 0]) / (2 * delta_eta)
        y_eta = (physical_plane[Mx, j+1, 1] - physical_plane[Mx, j-1, 1]) / (2 * delta_eta)
        
        x_eta_eta = (physical_plane[Mx, j+1, 0] - 2 * physical_plane[Mx, j, 0] + physical_plane[Mx, j-1, 0]) / (delta_eta * delta_eta)
        y_eta_eta = (physical_plane[Mx, j+1, 1] - 2 * physical_plane[Mx, j, 1] + physical_plane[Mx, j-1, 1]) / (delta_eta * delta_eta)
        
        # 边界一阶导数 (单侧差分，注意方向)
        x_xi_boundary = (physical_plane[Mx, j, 0] - physical_plane[Mx-1, j, 0]) / delta_xi
        y_xi_boundary = (physical_plane[Mx, j, 1] - physical_plane[Mx-1, j, 1]) / delta_xi
        
        # 计算度量系数
        alpha = x_eta * x_eta + y_eta * y_eta
        gamma = x_xi_boundary * x_xi_boundary + y_xi_boundary * y_xi_boundary
        
        # 正交投影计算
        dot_product = y_eta * x_xi_boundary - x_eta * y_xi_boundary
        x_xi_orthogonal = y_eta * dot_product / alpha
        y_xi_orthogonal = -x_eta * dot_product / alpha
        
        # Ghost点位置计算 (右边界，向外延伸)
        # x_{Mx+1,j} = x_{Mx,j} + (x_xi)_{Mx,j}
        ghost_x = physical_plane[Mx, j, 0] + x_xi_orthogonal * delta_xi
        ghost_y = physical_plane[Mx, j, 1] + y_xi_orthogonal * delta_xi
        
        # 二阶导数计算
        x_xi_xi = (physical_plane[Mx-1, j, 0] - 2 * physical_plane[Mx, j, 0] + ghost_x) / (delta_xi * delta_xi)
        y_xi_xi = (physical_plane[Mx-1, j, 1] - 2 * physical_plane[Mx, j, 1] + ghost_y) / (delta_xi * delta_xi)
        
        # 更新度量系数
        gamma_orthogonal = x_xi_orthogonal * x_xi_orthogonal + y_xi_orthogonal * y_xi_orthogonal
        
        # P、Q系数计算
        P[Mx, j] = -(x_xi_boundary * x_xi_xi + y_xi_boundary * y_xi_xi) / gamma_orthogonal - (x_xi_boundary * x_eta_eta + y_xi_boundary * y_eta_eta) / alpha
        Q[Mx, j] = -(x_eta * x_eta_eta + y_eta * y_eta_eta) / alpha - (x_eta * x_xi_xi + y_eta * y_xi_xi) / gamma_orthogonal
    '''
    # 底边界 (eta = 0)
    for i in range(1, Mx):
        # xi方向导数使用中心差分
        x_xi = (physical_plane[i+1, 0, 0] - physical_plane[i-1, 0, 0]) / (2 * delta_xi)
        y_xi = (physical_plane[i+1, 0, 1] - physical_plane[i-1, 0, 1]) / (2 * delta_xi)
        
        x_xi_xi = (physical_plane[i+1, 0, 0] - 2 * physical_plane[i, 0, 0] + physical_plane[i-1, 0, 0]) / (delta_xi * delta_xi)
        y_xi_xi = (physical_plane[i+1, 0, 1] - 2 * physical_plane[i, 0, 1] + physical_plane[i-1, 0, 1]) / (delta_xi * delta_xi)
        
        # 边界一阶导数 (单侧差分)
        x_eta_boundary = (physical_plane[i, 1, 0] - physical_plane[i, 0, 0]) / delta_eta
        y_eta_boundary = (physical_plane[i, 1, 1] - physical_plane[i, 0, 1]) / delta_eta
        
        # 计算度量系数
        gamma = x_xi * x_xi + y_xi * y_xi  # gamma
        alpha = x_eta_boundary * x_eta_boundary + y_eta_boundary * y_eta_boundary  # alpha_boundary
        
        # 正交投影计算 (基于公式 6.13)
        # x_eta = (-y_xi, x_xi)/g22 * (-y_xi·x_eta^0 + x_xi·y_eta^0)
        dot_product = -y_xi * x_eta_boundary + x_xi * y_eta_boundary
        x_eta_orthogonal = -y_xi * dot_product / gamma
        y_eta_orthogonal = x_xi * dot_product / gamma
        
        # Ghost点位置计算
        # x_{i,-1} = x_{i,0} - (x_eta)_{i,0}
        ghost_x = physical_plane[i, 0, 0] - x_eta_orthogonal * delta_eta
        ghost_y = physical_plane[i, 0, 1] - y_eta_orthogonal * delta_eta
        
        # 二阶导数计算
        x_eta_eta = (physical_plane[i, 1, 0] - 2 * physical_plane[i, 0, 0] + ghost_x) / (delta_eta * delta_eta)
        y_eta_eta = (physical_plane[i, 1, 1] - 2 * physical_plane[i, 0, 1] + ghost_y) / (delta_eta * delta_eta)
        
        # 更新度量系数
        alpha_orthogonal = x_eta_orthogonal * x_eta_orthogonal + y_eta_orthogonal * y_eta_orthogonal
        
        # P、Q系数计算
        P[i, 0] = -(x_xi * x_xi_xi + y_xi * y_xi_xi) / gamma - (x_xi * x_eta_eta + y_xi * y_eta_eta) / alpha_orthogonal
        Q[i, 0] = -(x_eta_boundary * x_eta_eta + y_eta_boundary * y_eta_eta) / alpha_orthogonal - (x_eta_boundary * x_xi_xi + y_eta_boundary * y_xi_xi) / gamma
    '''
    # 顶边界 (eta = My)
    for i in range(1, Mx):
        # xi方向导数使用中心差分
        x_xi = (physical_plane[i+1, My, 0] - physical_plane[i-1, My, 0]) / (2 * delta_xi)
        y_xi = (physical_plane[i+1, My, 1] - physical_plane[i-1, My, 1]) / (2 * delta_xi)
        
        x_xi_xi = (physical_plane[i+1, My, 0] - 2 * physical_plane[i, My, 0] + physical_plane[i-1, My, 0]) / (delta_xi * delta_xi)
        y_xi_xi = (physical_plane[i+1, My, 1] - 2 * physical_plane[i, My, 1] + physical_plane[i-1, My, 1]) / (delta_xi * delta_xi)
        
        # 边界一阶导数 (单侧差分)
        x_eta_boundary = (physical_plane[i, My, 0] - physical_plane[i, My-1, 0]) / delta_eta
        y_eta_boundary = (physical_plane[i, My, 1] - physical_plane[i, My-1, 1]) / delta_eta
        
        # 计算度量系数
        gamma = x_xi * x_xi + y_xi * y_xi
        alpha = x_eta_boundary * x_eta_boundary + y_eta_boundary * y_eta_boundary
        
        # 正交投影计算
        dot_product = -y_xi * x_eta_boundary + x_xi * y_eta_boundary
        x_eta_orthogonal = -y_xi * dot_product / gamma
        y_eta_orthogonal = x_xi * dot_product / gamma
        
        # Ghost点位置计算 (顶边界，向外延伸)
        # x_{i,My+1} = x_{i,My} + (x_eta)_{i,My}
        ghost_x = physical_plane[i, My, 0] + x_eta_orthogonal * delta_eta
        ghost_y = physical_plane[i, My, 1] + y_eta_orthogonal * delta_eta
        
        # 二阶导数计算
        x_eta_eta = (physical_plane[i, My-1, 0] - 2 * physical_plane[i, My, 0] + ghost_x) / (delta_eta * delta_eta)
        y_eta_eta = (physical_plane[i, My-1, 1] - 2 * physical_plane[i, My, 1] + ghost_y) / (delta_eta * delta_eta)
        
        # 更新度量系数
        alpha_orthogonal = x_eta_orthogonal * x_eta_orthogonal + y_eta_orthogonal * y_eta_orthogonal
        
        # P、Q系数计算
        P[i, My] = -(x_xi * x_xi_xi + y_xi * y_xi_xi) / gamma - (x_xi * x_eta_eta + y_xi * y_eta_eta) / alpha_orthogonal
        Q[i, My] = -(x_eta_boundary * x_eta_eta + y_eta_boundary * y_eta_eta) / alpha_orthogonal - (x_eta_boundary * x_xi_xi + y_eta_boundary * y_xi_xi) / gamma
    '''
    
    # TFI插值P、Q到内部点
    for i in range(1, Mx):
        for j in range(1,My):
            # 归一化坐标
            xi = i / Mx
            eta = j / My

            UP = (1 - xi) * P[0, j] + xi * P[Mx, j]
            VP = (1 - eta) * P[i, 0] + eta * P[i, My]
            UVP = (1-xi)*(1-eta)*P[0, 0] + xi*(1-eta)*P[Mx, 0] + \
                    (1-xi)*eta*P[0, My] + xi*eta*P[Mx, My]
            
            # P的双线性插值
            P[i, j] = (UP + VP)# - UVP)
            if i ==64:
                P[i,j] = 0
                #print(physical_plane[64,j])
                #input()
            if i==0:
                P[i,j] = 0
            
            UQ = (1 - xi) * Q[0, j] + xi * Q[Mx, j]
            VQ = (1 - eta) * Q[i, 0] + eta * Q[i, My]
            UVQ = (1-xi)*(1-eta)*Q[0, 0] + xi*(1-eta)*Q[Mx, 0] + \
                    (1-xi)*eta*Q[0, My] + xi*eta*Q[Mx, My]
            
            # Q的双线性插值  
            Q[i, j] = (UQ + VQ)# - UVQ)
            if i ==64:
                Q[i,j] = 0
            if i==0:
                Q[i,j] =0
    
    print(f"P range: [{np.min(P):.3e}, {np.max(P):.3e}]")
    print(f"Q range: [{np.min(Q):.3e}, {np.max(Q):.3e}]")
    
    return P, Q

def solve_elliptic_with_pq(physical_plane, P, Q, plot_iterations=True):
    """第二阶段：使用P、Q控制函数求解椭圆方程"""
    print("=== STAGE 2: Solving elliptic equations with P,Q control functions ===")
    
    max_iter = 2000
    tolerance = 1e-8
    new_plane = physical_plane.copy()

    # Create mask for boundary points (fixed)
    fixed_mask = np.zeros((Mx + 1, My + 1), dtype=bool)
    fixed_mask[:, 0] = True    # Inner boundary (airfoil)
    fixed_mask[:, My] = True   # Outer boundary (far-field)

    # Create directory for stage 2 plots
    if plot_iterations:
        plot_dir = "stage2_plots"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
            print(f"Created directory: {plot_dir}")

    for iteration in range(max_iter):
        max_error = 0.0

        # 每次迭代重新计算P、Q系数
        P, Q = calculate_pq_coefficients(physical_plane)
        
        for i in range(Mx):  # 修正：不使用周期性边界
            for j in range(1, My):
                
                #if fixed_mask[i, j]:
                #    continue
                
                # 标准索引（内部点不需要周期性边界）
                ip1 = (i + 1) % (Mx) #if i == Mx - 1 else i + 1
                im1 = (i - 1) % (Mx) #if i == 0 else i - 1
                
                # 一阶导数
                d_x_xi = (physical_plane[ip1, j, 0] - physical_plane[im1, j, 0]) / (2 * delta_xi)
                d_y_xi = (physical_plane[ip1, j, 1] - physical_plane[im1, j, 1]) / (2 * delta_xi)
                d_x_eta = (physical_plane[i, j+1, 0] - physical_plane[i, j-1, 0]) / (2 * delta_eta)
                d_y_eta = (physical_plane[i, j+1, 1] - physical_plane[i, j-1, 1]) / (2 * delta_eta)
                
                # 度量系数
                alpha = d_x_eta**2 + d_y_eta**2
                beta = d_x_xi * d_x_eta + d_y_xi * d_y_eta
                gamma = d_x_xi**2 + d_y_xi**2
                
                # 数值稳定性检查
                
                
                # 使用限制后的P、Q
                p = P[i,j]#np.clip(P[i, j], -5.0, 5.0)
                q = Q[i,j]#np.clip(Q[i, j], -5.0, 5.0)
                
                # 特殊处理中间线
            
                
                # 椭圆方程系数
                a11 = alpha / (delta_xi**2)
                a22 = gamma / (delta_eta**2)
                a12 = -beta / (2 * delta_xi * delta_eta)
                
                # 控制因子
                b = np.exp((-1/delta)*(i/Mx)*(j/My)*((Mx-i)/Mx)*((My-j)/My))
                
                # 交叉导数项
                cross_term_x = a12 * (physical_plane[ip1, j+1, 0] - physical_plane[ip1, j-1, 0] - 
                                     physical_plane[im1, j+1, 0] + physical_plane[im1, j-1, 0])
                cross_term_y = a12 * (physical_plane[ip1, j+1, 1] - physical_plane[ip1, j-1, 1] - 
                                     physical_plane[im1, j+1, 1] + physical_plane[im1, j-1, 1])
                
                # 分母
                denom = 2 * (a11 + a22)
                
                # 新坐标
                x_new = (a11 * (physical_plane[ip1, j, 0] + physical_plane[im1, j, 0]) +
                        a22 * (physical_plane[i, j+1, 0] + physical_plane[i, j-1, 0]) +
                        cross_term_x + alpha * p * d_x_xi * b +
                        gamma * q * d_x_eta * b) / denom
                
                y_new = (a11 * (physical_plane[ip1, j, 1] + physical_plane[im1, j, 1]) +
                        a22 * (physical_plane[i, j+1, 1] + physical_plane[i, j-1, 1]) +
                        cross_term_y + alpha * p * d_y_xi * b +
                        gamma * q * d_y_eta * b) / denom
                
               
                
                # 计算误差
                error_x = abs(x_new - physical_plane[i, j, 0])
                error_y = abs(y_new - physical_plane[i, j, 1])
                error = max(error_x, error_y)
                max_error = max(max_error, error)
                
                # 更新
                new_plane[i, j, 0] = x_new
                new_plane[i, j, 1] = y_new
        
        # 应用更新
        physical_plane = new_plane.copy()
        
        # Plot and save each iteration
        if plot_iterations and (iteration+1)%50==0:
            plt.figure(figsize=(10, 8))
            
            # Plot every 8th grid line for clarity and performance
            for j in range(0, My + 1, 2):
                plt.plot(physical_plane[:Mx, j, 0], physical_plane[:Mx, j, 1], 'b-', alpha=0.6, linewidth=0.5)

            # 绘制xi线（周向线）
            for i in range(0, Mx, 2):
                plt.plot(physical_plane[i, :, 0], physical_plane[i, :, 1], 'r-', alpha=0.6, linewidth=0.5)

            # 连接周期性边界：在x=0和x=Mx之间连线
            for j in range(0, My + 1, 2):
                # 连接最后一个点(Mx-1)和第一个点(0)
                x_connect = [physical_plane[Mx-1, j, 0], physical_plane[0, j, 0]]
                y_connect = [physical_plane[Mx-1, j, 1], physical_plane[0, j, 1]]
                plt.plot(x_connect, y_connect, 'b-', alpha=0.6, linewidth=0.5)
            # Highlight boundaries
            #plt.plot(physical_plane[:, 0, 0], physical_plane[:, 0, 1], 'k-', linewidth=2, label='Airfoil')
            #plt.plot(physical_plane[:, My, 0], physical_plane[:, My, 1], 'g-', linewidth=1, label='Far-field')
            
            plt.title(f'Stage 2: P,Q Grid - Iteration {iteration + 1:04d}, Error: {max_error:.2e}')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.axis('equal')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.xlim(-2, 2)
            plt.ylim(-2, 2)
            
            # Save plot with zero-padded iteration number
            plot_filename = os.path.join(plot_dir, f"stage2_iteration_{iteration+1:04d}.png")
            plt.savefig(plot_filename, dpi=100, bbox_inches='tight')
            
            # Close figure to prevent memory issues
            plt.close()
        
        # 打印进度
        if (iteration + 1) % 50 == 0:  # 更频繁的进度输出
            print(f"Stage 2 - Iteration {iteration + 1:04d}: max_error = {max_error:.2e}")
        
        # 检查收敛
        if max_error < tolerance:
            print(f"Stage 2 converged in {iteration + 1} iterations with error {max_error:.2e}")
            break
    
    if plot_iterations:
        print(f"All Stage 2 iteration plots saved in: {plot_dir}/")
    
    return physical_plane

def save_grid(physical_plane, filename):
    """保存网格到文件"""
    print(f"Saving grid to {filename}...")
    with open(filename, 'wb') as f:
        pickle.dump({
            'physical_plane': physical_plane,
            'Mx': Mx,
            'My': My,
            'delta_xi': delta_xi,
            'delta_eta': delta_eta
        }, f)
    print("Grid saved successfully!")

def load_grid(filename):
    """从文件加载网格"""
    print(f"Loading grid from {filename}...")
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    print("Grid loaded successfully!")
    return data['physical_plane']

def plot_final_grid(physical_plane, title="Final Grid", filename="final_grid.png"):
    """绘制最终网格"""
    plt.figure(figsize=(12, 8))

    # Plot grid lines
    for j in range(0, My + 1, 2):  # Plot every other eta line
        plt.plot(physical_plane[:, j, 0], physical_plane[:, j, 1], 'b-', alpha=0.6, linewidth=0.5)

    for i in range(0, Mx + 1, 2):  # Plot every other xi line
        plt.plot(physical_plane[i, :, 0], physical_plane[i, :, 1], 'r-', alpha=0.6, linewidth=0.5)

    # Highlight boundaries
    plt.plot(physical_plane[:, 0, 0], physical_plane[:, 0, 1], 'k-', linewidth=2, label='Airfoil')
    plt.plot(physical_plane[:, My, 0], physical_plane[:, My, 1], 'g-', linewidth=1, label='Far-field')

    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()

def calculate_grid_quality(physical_plane):
    """计算网格质量指标"""
    print("\nCalculating grid quality metrics...")
    
    # 网格正交性和纵横比
    orthogonality_errors = []
    aspect_ratios = []
    
    for i in range(1, Mx):
        for j in range(1, My):
            # xi方向向量
            xi_vec = np.array([
                physical_plane[i+1, j, 0] - physical_plane[i-1, j, 0],
                physical_plane[i+1, j, 1] - physical_plane[i-1, j, 1]
            ])
            
            # eta方向向量  
            eta_vec = np.array([
                physical_plane[i, j+1, 0] - physical_plane[i, j-1, 0],
                physical_plane[i, j+1, 1] - physical_plane[i, j-1, 1]
            ])
            
            # 正交性 (角度的余弦值，理想情况下应为0)
            xi_mag = np.linalg.norm(xi_vec)
            eta_mag = np.linalg.norm(eta_vec)
            
            if xi_mag > 1e-12 and eta_mag > 1e-12:
                cos_angle = np.dot(xi_vec, eta_vec) / (xi_mag * eta_mag)
                orthogonality_errors.append(abs(cos_angle))
                
                # 纵横比
                aspect_ratio = max(xi_mag, eta_mag) / min(xi_mag, eta_mag)
                aspect_ratios.append(aspect_ratio)
    
    if orthogonality_errors:
        print(f"Average orthogonality error: {np.mean(orthogonality_errors):.6f}")
        print(f"Max orthogonality error: {np.max(orthogonality_errors):.6f}")
        print(f"Average aspect ratio: {np.mean(aspect_ratios):.3f}")
        print(f"Max aspect ratio: {np.max(aspect_ratios):.3f}")
    else:
        print("No valid grid metrics calculated")

def main():
    """主函数 - 完整的两阶段网格生成流程"""
    print("=== AIRFOIL GRID GENERATION WITH P,Q CONTROL ===\n")
    
    # 文件名
    initial_grid_file = "initial_grid.pkl"
    final_grid_file = "final_pq_grid.pkl"
    
    # 第一阶段：生成或加载初始网格
    if not os.path.exists(initial_grid_file):
        print("=== STAGE 1: Creating initial grid ===")
        physical_plane = create_initial_grid()
        physical_plane = smooth_initial_grid(physical_plane, plot_iterations=False)
        save_grid(physical_plane, initial_grid_file)
        plot_final_grid(physical_plane, "Stage 1: Initial Smoothed Grid", "stage1_final.png")
    else:
        print("=== Loading existing initial grid ===")
        physical_plane = load_grid(initial_grid_file)
        plot_final_grid(physical_plane, "Loaded Initial Grid", "loaded_initial.png")
    
    print("\n" + "="*60)
    
    # 第二阶段：使用P、Q控制函数
    print("\n=== STAGE 2: Applying P,Q control functions ===")
    
    # 计算初始P、Q
    P, Q = calculate_pq_coefficients(physical_plane)
    
    # 求解带P、Q的椭圆方程
    final_grid = solve_elliptic_with_pq(physical_plane.copy(), P, Q, plot_iterations=True)
    
    # 保存最终网格
    save_grid(final_grid, final_grid_file)
    
    # 绘制最终结果
    plot_final_grid(final_grid, "Final Grid with P,Q Control", "final_pq_grid.png")
    
    # 计算网格质量
    calculate_grid_quality(final_grid)
    
    print(f"\n=== GRID GENERATION COMPLETED ===")
    print(f"Initial grid saved as: {initial_grid_file}")
    print(f"Final grid saved as: {final_grid_file}")
    print(f"Iteration plots saved in: stage2_plots/")
    print(f"\nTo create animation from plots:")
    print(f"ffmpeg -r 10 -pattern_type glob -i 'stage2_plots/*.png' -c:v libx264 -pix_fmt yuv420p grid_evolution.mp4")

if __name__ == "__main__":
    main()