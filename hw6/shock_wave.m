% Simplified Sod Shock Tube Problem - MATLAB Code
% Implementation with multiple numerical schemes

% Gas parameters and initial conditions
gamma = 1.4;  % Specific heat ratio
rho_L = 1.0; u_L = 0.0; p_L = 1.0;   % Left state
rho_R = 0.125; u_R = 0.0; p_R = 0.1;  % Right state

% Domain settings
x_min = -0.5; x_max = 0.5;
t_final = 0.25;

% Numerical scheme selection
% Options: 'rusanov', 'jst'
scheme = 'jst';  % Change this to select different schemes

% Number of grid points
nx = 100;
dx = (x_max - x_min) / nx;

% Calculate time step based on CFL condition
dt = 1e-5;
nt = floor(t_final / dt) + 1;
dt = t_final / nt;  % Adjust dt to match t_final exactly

% Initialize grid and solution arrays
x = linspace(x_min + dx/2, x_max - dx/2, nx);  % Cell centers
rho = zeros(1, nx);
rho_u = zeros(1, nx);
E = zeros(1, nx);

% Set initial conditions
for i = 1:nx
    if x(i) < 0
        rho(i) = rho_L;
        rho_u(i) = rho_L * u_L;
        E(i) = p_L / (gamma - 1) + 0.5 * rho_L * u_L^2;
    else
        rho(i) = rho_R;
        rho_u(i) = rho_R * u_R;
        E(i) = p_R / (gamma - 1) + 0.5 * rho_R * u_R^2;
    end
end

% Time integration loop using selected scheme
switch lower(scheme)
    case 'rusanov'
        % Rusanov scheme
        for n = 1:nt
            % Create copies for updating
            rho_new = rho;
            rho_u_new = rho_u;
            E_new = E;
            
            for i = 2:nx-1
                % Calculate primitive variables
                u_im1 = rho_u(i-1) / rho(i-1);
                u_i = rho_u(i) / rho(i);
                u_ip1 = rho_u(i+1) / rho(i+1);
                
                p_im1 = (gamma - 1) * (E(i-1) - 0.5 * rho(i-1) * u_im1^2);
                p_i = (gamma - 1) * (E(i) - 0.5 * rho(i) * u_i^2);
                p_ip1 = (gamma - 1) * (E(i+1) - 0.5 * rho(i+1) * u_ip1^2);
                
                % Calculate sound speeds
                a_im1 = sqrt(gamma * p_im1 / rho(i-1));
                a_i = sqrt(gamma * p_i / rho(i));
                a_ip1 = sqrt(gamma * p_ip1 / rho(i+1));
                
                % Calculate fluxes
                F_im1 = [rho_u(i-1); 
                         rho_u(i-1) * u_im1 + p_im1; 
                         (E(i-1) + p_im1) * u_im1];
                
                F_i = [rho_u(i); 
                       rho_u(i) * u_i + p_i; 
                       (E(i) + p_i) * u_i];
                
                F_ip1 = [rho_u(i+1); 
                         rho_u(i+1) * u_ip1 + p_ip1; 
                         (E(i+1) + p_ip1) * u_ip1];
                
                % Calculate local wave speeds
                lambda_im = max(abs(u_i) + a_i, abs(u_im1) + a_im1);
                lambda_ip = max(abs(u_i) + a_i, abs(u_ip1) + a_ip1);
                
                % Rusanov numerical fluxes
                F_num_im = 0.5 * (F_im1 + F_i) - 0.5 * lambda_im * ([rho(i); rho_u(i); E(i)] - [rho(i-1); rho_u(i-1); E(i-1)]);
                F_num_ip = 0.5 * (F_i + F_ip1) - 0.5 * lambda_ip * ([rho(i+1); rho_u(i+1); E(i+1)] - [rho(i); rho_u(i); E(i)]);
                
                % Update solution
                rho_new(i) = rho(i) - dt/dx * (F_num_ip(1) - F_num_im(1));
                rho_u_new(i) = rho_u(i) - dt/dx * (F_num_ip(2) - F_num_im(2));
                E_new(i) = E(i) - dt/dx * (F_num_ip(3) - F_num_im(3));
            end
            
            % Apply boundary conditions
            rho_new(1) = rho_new(2);
            rho_u_new(1) = rho_u_new(2);
            E_new(1) = E_new(2);
            
            rho_new(nx) = rho_new(nx-1);
            rho_u_new(nx) = rho_u_new(nx-1);
            E_new(nx) = E_new(nx-1);
            
            % Update values for next time step
            rho = rho_new;
            rho_u = rho_u_new;
            E = E_new;
        end
        
    case 'jst'
        % JST (Jameson-Schmidt-Turkel) scheme with adaptive artificial dissipation
        % Parameters for JST scheme
        k_2 = 0.55;  % Second-order dissipation coefficient
        k_4 = 1.5/128;  % Fourth-order dissipation coefficient
        
        for n = 1:nt
            % Create copies for updating
            rho_new = rho;
            rho_u_new = rho_u;
            E_new = E;
            
            for i = 4:nx-3  % Need wider stencil for JST
                % Calculate primitive variables
                u_im3 = rho_u(i-3) / rho(i-3);
                u_im2 = rho_u(i-2) / rho(i-2);
                u_im1 = rho_u(i-1) / rho(i-1);
                u_i = rho_u(i) / rho(i);
                u_ip1 = rho_u(i+1) / rho(i+1);
                u_ip2 = rho_u(i+2) / rho(i+2);
                u_ip3 = rho_u(i+3) / rho(i+3);

                p_im3 = (gamma - 1) * (E(i-3) - 0.5 * rho(i-3) * u_im3^2);
                p_im2 = (gamma - 1) * (E(i-2) - 0.5 * rho(i-2) * u_im2^2);
                p_im1 = (gamma - 1) * (E(i-1) - 0.5 * rho(i-1) * u_im1^2);
                p_i = (gamma - 1) * (E(i) - 0.5 * rho(i) * u_i^2);
                p_ip1 = (gamma - 1) * (E(i+1) - 0.5 * rho(i+1) * u_ip1^2);
                p_ip2 = (gamma - 1) * (E(i+2) - 0.5 * rho(i+2) * u_ip2^2);
                p_ip3 = (gamma - 1) * (E(i+3) - 0.5 * rho(i+3) * u_ip3^2);
                
                % Calculate sound speeds
                a_im1 = sqrt(gamma * p_im1 / rho(i-1));
                a_i = sqrt(gamma * p_i / rho(i));
                a_ip1 = sqrt(gamma * p_ip1 / rho(i+1));
                
                % Calculate fluxes
                F_im1 = [rho_u(i-1); 
                         rho_u(i-1) * u_im1 + p_im1; 
                         (E(i-1) + p_im1) * u_im1];
                
                F_i = [rho_u(i); 
                       rho_u(i) * u_i + p_i; 
                       (E(i) + p_i) * u_i];
                
                F_ip1 = [rho_u(i+1); 
                         rho_u(i+1) * u_ip1 + p_ip1; 
                         (E(i+1) + p_ip1) * u_ip1];

                % Conservative variables vectors for dissipation terms
                U_im3 = [rho(i-3); rho_u(i-3); E(i-3)];
                U_im2 = [rho(i-2); rho_u(i-2); E(i-2)];
                U_im1 = [rho(i-1); rho_u(i-1); E(i-1)];
                U_i = [rho(i); rho_u(i); E(i)];
                U_ip1 = [rho(i+1); rho_u(i+1); E(i+1)];
                U_ip2 = [rho(i+2); rho_u(i+2); E(i+2)];
                U_ip3 = [rho(i+3); rho_u(i+3); E(i+3)];
                
                % Calculate local wave speeds
                lambda_im = max(abs(u_i) + a_i, abs(u_im1) + a_im1);
                lambda_ip = max(abs(u_i) + a_i, abs(u_ip1) + a_ip1);
                
                % Calculate pressure sensors for adaptive dissipation
                nu_im2 = abs((p_im1 - 2*p_im2 + p_im3)) ./ abs(p_im1 + 2*p_im2 + p_im3 + eps);
                nu_im1 = abs((p_i - 2*p_im1 + p_im2)) ./ abs(p_i + 2*p_im1 + p_im2 + eps);
                nu_i = abs((p_ip1 - 2*p_i + p_im1)) ./ abs(p_ip1 + 2*p_i + p_im1 + eps);
                nu_ip1 = abs((p_ip2 - 2*p_ip1 + p_i)) ./ abs(p_ip2 + 2*p_ip1 + p_i + eps);
                nu_ip2 = abs((p_ip3 - 2*p_ip2 + p_ip1)) ./ abs(p_ip3 + 2*p_ip2 + p_ip1 + eps);

                % Adaptive coefficients
                eps2_im = k_2 * max([nu_im2, nu_im1, nu_i, nu_ip1]);
                eps2_ip = k_2 * max([nu_im1, nu_i, nu_ip1, nu_ip2]);
                
                eps4_im = max(0, k_4 - eps2_im);
                eps4_ip = max(0, k_4 - eps2_ip);

                % Central fluxes with artificial dissipation (JST scheme)
                F_num_im = 0.5 * (F_im1 + F_i) - eps2_im * lambda_im * (U_i - U_im1) + eps4_im * lambda_im * (U_ip1 - 3*U_i + 3*U_im1 - U_im2);
                F_num_ip = 0.5 * (F_i + F_ip1) - eps2_ip * lambda_ip * (U_ip1 - U_i) + eps4_ip * lambda_ip * (U_ip2 - 3*U_ip1 + 3*U_i - U_im1);
                
                % Update solution
                rho_new(i) = rho(i) - dt/dx * (F_num_ip(1) - F_num_im(1));
                rho_u_new(i) = rho_u(i) - dt/dx * (F_num_ip(2) - F_num_im(2));
                E_new(i) = E(i) - dt/dx * (F_num_ip(3) - F_num_im(3));
            end
            
            % Update boundary regions (simple extrapolation for JST scheme)
            for i = 1:3
                rho_new(i) = rho_new(4);
                rho_u_new(i) = rho_u_new(4);
                E_new(i) = E_new(4);
                
                rho_new(nx-i+1) = rho_new(nx-3);
                rho_u_new(nx-i+1) = rho_u_new(nx-3);
                E_new(nx-i+1) = E_new(nx-3);
            end
            
            % Update values for next time step
            rho = rho_new;
            rho_u = rho_u_new;
            E = E_new;
        end
    otherwise
        error('Unknown scheme: %s. Available options: rusanov, jst', scheme);
end

% Calculate primitive variables for plotting
u = rho_u ./ rho;
p = (gamma - 1) * (E - 0.5 * rho .* u.^2);
file = "rus"+string(nx)+string(dt);
% Create separate figures for density, velocity and pressure
% Figure 1: Density
figure('Position', [100, 100, 600, 400]);
plot(x, rho, 'b-', 'LineWidth', 2);
xlabel('Position (x)');
ylabel('\rho');
grid on;
saveas(gcf, file + "_density.png");
% Figure 2: Velocity
figure('Position', [100, 550, 600, 400]);
plot(x, u, 'r-', 'LineWidth', 2);
xlabel('Position (x)');
ylabel('u');
ylim([-0.2, 1.2]);
grid on;
saveas(gcf, file + "_u.png");
% Figure 3: Pressure
figure('Position', [750, 100, 600, 400]);
plot(x, p, 'g-', 'LineWidth', 2);
xlabel('Position (x)');
ylabel('p');
grid on;
saveas(gcf, file + "_p.png");