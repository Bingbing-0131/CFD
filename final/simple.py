import numpy as np
import matplotlib.pyplot as plt
import glob
# 网格大小和模拟参数
nx = 128   # 网格点数（x方向）
ny = 128   # 网格点数（y方向）
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
nt = 2000    # 时间步数
u_top = 1.0  # 顶部边界的速度

Re = 400

# 边界条件初始化 - CORRECTED
def apply_boundary_conditions():
    u[:, -1] = u_top  
    u[:, 0] = 0      
    u[0, :] = 0       
    u[-1, :] = 0  
    
        
    
    v[:, 0] = 0       
    v[:, -1] = 0      
    v[0, :] = 0       
    v[-1, :] = 0      
    

def estimate_pressure():
    p_est[:, :] = p[:, :]

def compute_param_u():
    for i in range(1, nx):  # 修正索引范围
        for j in range(0, ny ):
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
    for i in range(0, nx):
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
    
    for i in range(1, nx-1):
        param_p[i, 0, 0] = (dy * dy / param_u[i+1, 0, 0] + 
                               dy * dy / param_u[i, 0, 0] + 
                               dx * dx / param_v[i, 1, 0] )
        param_p[i, 0, 1] = dy * dy / param_u[i+1, 0, 0]
        param_p[i, 0, 2] = dy * dy / param_u[i, 0, 0]
        param_p[i, 0, 3] = dx * dx / param_v[i, 1, 0]
        param_p[i, 0, 4] = 0.0
            # 连续性方程的源项
        param_p[i, 0, 5] = -rho * dx * dy * ((u_est[i+1, 0] - u_est[i, 0]) / dx + 
                                               (v_est[i, 1] - v_est[i, 0]) / dy)
        
        param_p[i, ny-1, 0] = (dy * dy / param_u[i+1, ny-1, 0] + 
                               dy * dy / param_u[i, ny-1, 0] + 
                               dx * dx / param_v[i, ny-1, 0])
        param_p[i, ny-1, 1] = dy * dy / param_u[i+1, ny-1, 0]
        param_p[i, ny-1, 2] = dy * dy / param_u[i, ny-1, 0]
        param_p[i, ny-1, 3] = 0.0
        param_p[i, ny-1, 4] = dx * dx / param_v[i, ny-1, 0]
            # 连续性方程的源项
        param_p[i, ny-1, 5] = -rho * dx * dy * ((u_est[i+1, ny-1] - u_est[i, ny-1]) / dx + 
                                               (v_est[i, ny] - v_est[i, ny-1]) / dy)
    param_p[0,0,0] = param_p[1,0,0]
    param_p[0,0,1] = param_p[1,0,1]
    param_p[0,0,2] = param_p[1,0,2]
    param_p[0,0,3] = param_p[1,0,3]
    param_p[0,0,4] = param_p[1,0,4]

    param_p[0,ny-1,0] = param_p[1,ny-1,0]
    param_p[0,ny-1,1] = param_p[1,ny-1,1]
    param_p[0,ny-1,2] = param_p[1,ny-1,2]
    param_p[0,ny-1,3] = param_p[1,ny-1,3]
    param_p[0,ny-1,4] = param_p[1,ny-1,4]

    param_p[nx-1,0,0] = param_p[nx-2,0,0]
    param_p[nx-1,0,1] = param_p[nx-2,0,1]
    param_p[nx-1,0,2] = param_p[nx-2,0,2]
    param_p[nx-1,0,3] = param_p[nx-2,0,3]
    param_p[nx-1,0,4] = param_p[nx-2,0,4]

    param_p[nx-1,ny-1,0] = param_p[nx-2,ny-1,0]
    param_p[nx-1,ny-1,1] = param_p[nx-2,ny-1,1]
    param_p[nx-1,ny-1,2] = param_p[nx-2,ny-1,2]
    param_p[nx-1,ny-1,3] = param_p[nx-2,ny-1,3]
    param_p[nx-1,ny-1,4] = param_p[nx-2,ny-1,4]

    for j in range(1, ny-1):
        param_p[0, j, 0] = (dy * dy / param_u[1, j, 0] + 
                               dx * dx / param_v[0, j+1, 0] + 
                               dx * dx / param_v[0, j, 0])
        param_p[0, j, 1] = dy * dy / param_u[1, j, 0]
        param_p[0, j, 2] = 0.0
        param_p[0, j, 3] = dx * dx / param_v[0, j+1, 0]
        param_p[0, j, 4] = dx * dx / param_v[0, j, 0]
            # 连续性方程的源项
        param_p[0, j, 5] = -rho * dx * dy * ((u_est[1, j] - u_est[0, j]) / dx + 
                                               (v_est[0, j+1] - v_est[0, j]) / dy)
        

        param_p[nx-1, j, 0] = (dy * dy / param_u[nx-1, j, 0]+
                               dx * dx / param_v[nx-1, j+1, 0] + 
                               dx * dx / param_v[nx-1, j, 0])
        param_p[nx-1, j, 1] = 0.0
        param_p[nx-1, j, 2] = dy * dy /param_u[nx-1, j, 0]
        param_p[nx-1, j, 3] = dx * dx / param_v[nx-1, j+1, 0]
        param_p[nx-1, j, 4] = dx * dx / param_v[nx-1, j, 0]
            # 连续性方程的源项
        param_p[nx-1, j, 5] = -rho * dx * dy * ((u_est[nx, j] - u_est[nx-1, j]) / dx + 
                                               (v_est[nx-1, j+1] - v_est[nx-1, j]) / dy)
    #print(param_u[64,62,0])


def Solve_Poisson_p():
    p_correct.fill(0.0)
    for it in range(max_iter):
        p_correct_save[:, :] = p_correct[:, :]
        for i in range(0, nx):
            for j in range(0, ny):
                if abs(param_p[i, j, 0]) > 1e-12:  # 避免除零
                    ip = i+1 if i<nx-1 else nx-1
                    im = i-1 if i>0 else 0
                    ir = j+1 if j<ny-1 else ny-1
                    il = j-1 if j>0 else 0
                    p_correct[i, j] = (param_p[i, j, 1] * p_correct_save[ip, j] + 
                                      param_p[i, j, 2] * p_correct_save[im, j] + 
                                      param_p[i, j, 3] * p_correct_save[i, ir] + 
                                      param_p[i, j, 4] * p_correct_save[i, il] + 
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

def load_latest_results():
    """加载最新的计算结果"""
    global u, v, p
    
    # 查找所有保存的u和v文件
    u_files = glob.glob('u_*.npy')
    v_files = glob.glob('v_*.npy')
    
    if not u_files or not v_files:
        print("未找到保存的结果文件，将从初始状态开始")
        return 0
    
    # 获取最新的文件（最大的时间步）
    u_files.sort()
    v_files.sort()
    
    latest_u_file = u_files[-1]
    latest_v_file = v_files[-1]
    
    # 从文件名中提取时间步数
    import re
    u_step = int(re.findall(r'u_(\d+)\.npy', latest_u_file)[0])
    v_step = int(re.findall(r'v_(\d+)\.npy', latest_v_file)[0])
    
    if u_step != v_step:
        print(f"警告：u和v文件的时间步不匹配 (u: {u_step}, v: {v_step})")
        start_step = min(u_step, v_step)
        latest_u_file = f'u_{start_step:04d}.npy'
        latest_v_file = f'v_{start_step:04d}.npy'
    else:
        start_step = u_step
    
    # 加载数据
    try:
        u = np.load(latest_u_file)
        v = np.load(latest_v_file)
        print(f"成功加载第 {start_step} 步的结果")
        print(f"u 形状: {u.shape}, v 形状: {v.shape}")
        print(f"u 范围: [{u.min():.6f}, {u.max():.6f}]")
        print(f"v 范围: [{v.min():.6f}, {v.max():.6f}]")
        
        # 检查数据完整性
        if u.shape != (nx+1, ny) or v.shape != (nx, ny+1):
            print(f"错误：加载的数据形状不正确")
            print(f"期望 u: ({nx+1}, {ny}), v: ({nx}, {ny+1})")
            print(f"实际 u: {u.shape}, v: {v.shape}")
            return 0
            
        return start_step
        
    except Exception as e:
        print(f"加载文件时出错: {e}")
        return 0

def plot_results(n=None):
    """绘制速度流线图（streamplot）"""
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

    # 创建图像和坐标轴
    fig, ax = plt.subplots(figsize=(8, 6))  # 正确方式

    # 绘制速度流线图
    ax.streamplot(X, Y, u_center.T, v_center.T, density=1.2, linewidth=1)

    ax.set_title('Velocity Streamlines')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    plt.tight_layout()

    if n is not None:
        plt.savefig(f"stream_{n:04d}.png")
        plt.close()
    else:
        plt.show()


saved_results = []
def continue_cavity_flow(additional_steps=1000):
    """继续求解方腔流"""
    global u, v, p
    
    # 加载之前的结果
    start_step = load_latest_results()
    
    if start_step == 0:
        print("从初始状态开始计算...")
        # 初始化边界条件
        apply_boundary_conditions()
    else:
        print(f"从第 {start_step} 步继续计算...")
        # 应用边界条件以确保边界正确
        apply_boundary_conditions()
    
    print(f"将继续计算 {additional_steps} 步...")
    
    # 继续计算
    for n in range(start_step + 1, start_step + additional_steps + 1):
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

        # 保存结果和检查收敛性
        if n % 50 == 0:
            plot_results(n)
            np.save(f'u_{n:04d}.npy', u)
            np.save(f'v_{n:04d}.npy', v)
            print(f"时间步 {n} 已保存")

        if n % 100 == 0:
            div_max = check_convergence()
            print(f"时间步 {n}, 最大散度: {div_max:.6f}")
            if div_max < 1e-6:
                print(f"在第 {n} 步收敛")
                break

    print("计算完成！")
    
    # 绘制最终结果
    plot_results()
    
    return start_step + additional_steps

def analyze_results():
    """分析计算结果"""
    div_max = check_convergence()
    print(f"\n=== 结果分析 ===")
    print(f"最大散度: {div_max:.8f}")
    print(f"u速度范围: [{u.min():.6f}, {u.max():.6f}]")
    print(f"v速度范围: [{v.min():.6f}, {v.max():.6f}]")
    print(f"压力范围: [{p.min():.6f}, {p.max():.6f}]")
    
    # 计算中心线速度分布
    u_centerline = np.zeros(ny)
    v_centerline = np.zeros(nx)
    
    for j in range(ny):
        u_centerline[j] = (u[nx//2, j] + u[nx//2 + 1, j]) / 2
    
    for i in range(nx):
        v_centerline[i] = (v[i, ny//2] + v[i, ny//2 + 1]) / 2
    
    print(f"中心线最大u速度: {u_centerline.max():.6f}")
    print(f"中心线最大v速度: {abs(v_centerline).max():.6f}")

if __name__ == "__main__":
    # 继续计算更多步骤
    final_step = continue_cavity_flow(additional_steps=1000)
    
    # 分析结果
    analyze_results()
    
    print(f"\n计算已完成到第 {final_step} 步")
    print("可以通过调用 continue_cavity_flow(additional_steps=N) 继续计算更多步骤")