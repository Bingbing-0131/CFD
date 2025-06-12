import numpy as np
import matplotlib.pyplot as plt

# Grid resolution
Mx = 128
My = 128
delta_eta = 1.0 / Mx
delta_xi = 1.0 / My

delta = 0.05

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

# Initialize interior points with linear interpolation
for i in range(Mx + 1):
    for j in range(1, My):
        t = j / My
        physical_plane[i, j, 0] = (1-t) * physical_plane[i, 0, 0] + t * physical_plane[i, My, 0]
        physical_plane[i, j, 1] = (1-t) * physical_plane[i, 0, 1] + t * physical_plane[i, My, 1]

# Jacobi iteration for grid smoothing
max_iter = 5000
tolerance = 1e-8
new_plane = physical_plane.copy()

# Create mask for boundary points (fixed)
fixed_mask = np.zeros((Mx + 1, My + 1), dtype=bool)
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

            b = np.exp((-1/delta)*(i/Mx)*(j/My)*((Mx-i)/Mx)*((My-j)/My))
            
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

# Plotting
plt.figure(figsize=(12, 8))

# Plot grid lines
for j in range(0, My + 1, 2):  # Plot every other eta line
    plt.plot(physical_plane[:, j, 0], physical_plane[:, j, 1], 'b-', alpha=0.6, linewidth=0.5)

for i in range(0, Mx + 1, 2):  # Plot every other xi line
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