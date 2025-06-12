import numpy as np
import matplotlib.pyplot as plt

# 网格大小和模拟参数
nx = 64   # 网格点数（x方向）
ny = 64   # 网格点数（y方向）
max_iter = 100
u = np.zeros((nx+1, ny))
v = np.zeros((nx,ny+1))
p = np.zeros((nx,ny))

u_est = np.zeros((nx+1, ny))
v_est = np.zeros((nx,ny+1))
p_est = np.zeros((nx,ny))

u_save = np.zeros((nx+1,ny))
v_save = np.zeros((nx,ny+1))
p_save = np.zeros((nx,ny))

p_correct = np.zeros((nx,ny))
p_correct_save = np.zeros((nx,ny))

param_u = np.zeros((nx+1,ny,6)) 
param_v = np.zeros((nx,ny+1,6))
param_p = np.zeros((nx,ny,6))

alpha_u = np.zeros((nx,ny))
alpha_v = np.zeros((nx+1,ny+1))


beta_u = np.zeros((nx + 1, ny + 1))

beta_v = np.zeros((nx, ny))

lx = 1.0  # x方向的长度
ly = 1.0  # y方向的长度
dx = lx / nx  # 网格步长（x方向）
dy = ly / ny  # 网格步长（y方向）

# 流体参数
rho = 1.0    # 密度
nu = 0.01    # 动力粘度
dt = 0.001   # 时间步长
nt = 1000    # 时间步数
u_top = 1.0  # 顶部边界的速度

Re = 400

# 边界条件初始化 - CORRECTED
def apply_boundary_conditions():
    u[:, 0] = 0      
    u[0, :] = 0       
    u[-1, :] = 0  
    u[:, -1] = u_top  
        
    
    v[:, 0] = 0       
    v[:, -1] = 0      
    v[0, :] = 0       
    v[-1, :] = 0      
    

def estimate_pressure():
    p_est[:, :] = p[:, :]

def compute_param_u():
    for i in range(1, nx):  # 修正索引范围
        for j in range(1, ny - 1):
            param_u[i, j, 0] = (dx * dy / dt + 
                               dy * ((alpha_u[i, j] - alpha_u[i-1, j])/2 + 2 /(Re* dx)) + 
                               dx * ((alpha_v[i, j+1] - alpha_v[i, j])/2 + 2 /(Re* dx)))
            param_u[i, j, 1] = dy * (alpha_u[i-1, j]/2 + 1 /(Re* dx))
            param_u[i, j, 2] = dy * (-alpha_u[i, j]/2 + 1 /(Re* dx))
            param_u[i, j, 3] = dx * (alpha_v[i, j]/2 + 1 /(Re* dy))
            param_u[i, j, 4] = dx * (-alpha_v[i, j+1]/2 + 1 /(Re* dy))
            param_u[i, j, 5] = (-dy * (p_est[i, j] - p_est[i-1, j]) + 
                               dx * dy / dt * u[i, j])

def compute_param_v():
    for i in range(1, nx-1):
        for j in range(1, ny):  # 修正索引范围
            param_v[i, j, 0] = (dx * dy / dt + 
                               dx * ((beta_v[i,j] -beta_v[i,j-1])/2 + 2 /(Re* dx)) + 
                               dy * ((beta_u[i+1, j] - beta_u[i, j])/2 + 2 /(Re* dx)))
            param_v[i, j, 1] = dx * (beta_v[i, j-1]/2 + 1 /(Re* dx))
            param_v[i, j, 2] = dx * (-beta_v[i, j]/2 + 1 /(Re* dx))
            param_v[i, j, 3] = dy * (beta_u[i, j]/2 + 1 /(Re* dx))
            param_v[i, j, 4] = dy * (-beta_u[i+1, j]/2 + 1 /(Re* dx))
            param_v[i, j, 5] = (-dx * (p_est[i, j] - p_est[i, j-1]) + 
                               dx * dy / dt * v[i, j])

def compute_alpha():
    for i in range(nx):
        for j in range(ny):
            alpha_u[i,j] = (u[i,j]+u[i+1,j])/2

    for i in range(1,nx):
        for j in range(ny+1):
            alpha_v[i,j] = (v[i-1,j] +v[i,j])/2
    alpha_v[0,:] = 0
    alpha_v[nx,:] = 0
            

def compute_beta():
    for i in range(nx):
        for j in range(ny):
            beta_v[i,j] = (v[i,j] +v[i,j+1])/2
    
    
    for i in range(nx+1):
        for j in range(1,ny):
            beta_u[i,j] = (u[i,j-1] + u[i,j])/2
            #print(beta_u[i,j])
    beta_u[:,ny] = 1.0
    beta_u[:,0] = 0.0
               

def Solve_Poisson_u(): 
    u_est[:, :] = u[:, :]
    for it in range(max_iter):
        u_save[:, :] = u_est[:, :]
        for i in range(1, nx):
            for j in range(1, ny - 1):
                if abs(param_u[i, j, 0]) > 1e-12:  # 避免除零
                    u_est[i, j] = (param_u[i, j, 1] * u_save[i-1, j] + 
                                  param_u[i, j, 2] * u_save[i+1, j] + 
                                  param_u[i, j, 3] * u_save[i, j-1] + 
                                  param_u[i, j, 4] * u_save[i, j+1] + 
                                  param_u[i, j, 5]) / param_u[i, j, 0]

def Solve_Poisson_v():
    v_est[:, :] = v[:, :]
    for it in range(max_iter):
        v_save[:, :] = v_est[:, :]
        for i in range(1, nx-1):
            for j in range(1, ny):
                if abs(param_v[i, j, 0]) > 1e-12:  # 避免除零
                    v_est[i, j] = (param_v[i, j, 1] * v_save[i, j-1] + 
                                  param_v[i, j, 2] * v_save[i, j+1] + 
                                  param_v[i, j, 3] * v_save[i-1, j] + 
                                  param_v[i, j, 4] * v_save[i+1, j] + 
                                  param_v[i, j, 5]) / param_v[i, j, 0]

def compute_param_p():
    for i in range(1, nx-1):
        for j in range(1, ny - 1):
            # 计算压力修正方程的系数
            param_p[i, j, 0] = (dy * dy / param_u[i+1, j, 0] + 
                               dy * dy / param_u[i, j, 0] + 
                               dx * dx / param_v[i, j+1, 0] + 
                               dx * dx / param_v[i, j, 0])
            param_p[i, j, 1] = dy * dy / param_u[i+1, j, 0]
            param_p[i, j, 2] = dy * dy / param_u[i, j, 0]
            param_p[i, j, 3] = dx * dx / param_v[i, j+1, 0]
            param_p[i, j, 4] = dx * dx / param_v[i, j, 0]
            # 连续性方程的源项
            param_p[i, j, 5] = -rho * dx * dy * ((u_est[i+1, j] - u_est[i, j]) / dx + 
                                               (v_est[i, j+1] - v_est[i, j]) / dy)
    #print(param_u[64,62,0])


def Solve_Poisson_p():
    p_correct.fill(0.0)
    for it in range(max_iter):
        p_correct_save[:, :] = p_correct[:, :]
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                if abs(param_p[i, j, 0]) > 1e-12:  # 避免除零
                    p_correct[i, j] = (param_p[i, j, 1] * p_correct_save[i + 1, j] + 
                                      param_p[i, j, 2] * p_correct_save[i - 1, j] + 
                                      param_p[i, j, 3] * p_correct_save[i, j + 1] + 
                                      param_p[i, j, 4] * p_correct_save[i, j - 1] + 
                                      param_p[i, j, 5]) / param_p[i, j, 0]
        #print(p_correct[61,61],param_p[61,61,1],param_p[61,61,2],param_p[61,61,3],param_p[61,61,4],param_p[61,61,5],param_p[61,61,0])

def correct_velocity_pressure():
    # 修正u速度
    for i in range(1, nx):
        for j in range(1, ny-1):
            if abs(param_u[i, j, 0]) > 1e-12:
                u[i, j] = u_est[i, j] - dy / param_u[i, j, 0] * (p_correct[i, j] - p_correct[i-1, j])
                if u[i,j]>1.0:
                    print(u[i,j],i,j)
    print(p_correct[63,62],p_correct[62,62])
    # 修正v速度  
    for i in range(1, nx-1):
        for j in range(1, ny):
            if abs(param_v[i, j, 0]) > 1e-12:
                v[i, j] = v_est[i, j] - dx / param_v[i, j, 0] * (p_correct[i, j] - p_correct[i, j-1])
    
    # 修正压力
    alpha_p = 0.8  # 压力松弛因子
    p[:, :] = p_est[:, :] + p_correct[:, :]

def check_convergence():
    """检查收敛性"""
    div_max = 0.0
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            div_u = (u[i+1, j] - u[i, j]) / dx + (v[i, j+1] - v[i, j]) / dy
            div_max = max(div_max, abs(div_u))
    return div_max

def plot_results(n=None):
    """绘制结果"""
    # 创建网格用于绘图
    x = np.linspace(0, lx, nx)
    y = np.linspace(0, ly, ny)
    X, Y = np.meshgrid(x, y)
    
    # 插值速度到网格中心
    u_center = np.zeros((nx, ny))
    v_center = np.zeros((nx, ny))
    
    for i in range(nx):
        for j in range(ny):
            u_center[i, j] = (u[i, j] + u[i+1, j]) / 2
            v_center[i, j] = (v[i, j] + v[i, j+1]) / 2
    
    # 绘制结果
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 压力场
    im1 = ax1.contourf(X, Y, p.T, levels=20, cmap='viridis')
    ax1.set_title('Pressure Field')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    plt.colorbar(im1, ax=ax1)
    
    # 速度矢量场
    skip = 4
    ax2.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
               u_center[::skip, ::skip].T, v_center[::skip, ::skip].T)
    ax2.set_title('Velocity Vector Field')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_aspect('equal')
    
    # u速度等值线
    im3 = ax3.contourf(X, Y, u_center.T, levels=20, cmap='coolwarm')
    ax3.set_title('u-velocity')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    plt.colorbar(im3, ax=ax3)
    
    # v速度等值线
    im4 = ax4.contourf(X, Y, v_center.T, levels=20, cmap='coolwarm')
    ax4.set_title('v-velocity')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    plt.colorbar(im4, ax=ax4)
    
    plt.tight_layout()
    if n is not None:
        plt.savefig(f"test_{n:04d}.png")
    else:
        plt.show()

# 主求解循环
def solve_cavity_flow():
    """主求解函数 - 修正版本"""
    print("开始求解方腔流...")
    
    # 初始化边界条件
    apply_boundary_conditions()
    
    for n in range(nt):
        if n % 50 == 0:
            plot_results(n)
            print(f"时间步: {n}")
        
        # SIMPLE算法步骤
        estimate_pressure()
        compute_alpha()
        compute_beta()
        compute_param_u()
        compute_param_v()
        
        Solve_Poisson_u()
        Solve_Poisson_v()
        
        compute_param_p()
        Solve_Poisson_p()
        
        correct_velocity_pressure()
        apply_boundary_conditions()

        
        
        # 检查收敛性
        if n % 100 == 0:
            div_max = check_convergence()
            print(f"时间步 {n}, 最大散度: {div_max:.6f}")
            
            if div_max < 1e-6:
                print(f"在第 {n} 步收敛")
                break
    
    print("求解完成！")
    plot_results()

# 运行求解
if __name__ == "__main__":
    solve_cavity_flow()