gamma = 1.4; 
rho_L = 5.0; u_L = sqrt(1.4); p_L = 29.0;   
rho_R = 1.0; u_R = 5*sqrt(1.4); p_R = 1.0;

x_min = 0; x_max = 1.0;
t_final = 0.21;
epsilon = 0.1;
scheme = 'roe';  

nx = 200;
dx = (x_max - x_min) / nx;

dt = 1e-5;
nt = floor(t_final / dt) + 1;
dt = t_final / nt; 

x = linspace(x_min + dx/2, x_max - dx/2, nx);  
rho = zeros(1, nx);
rho_u = zeros(1, nx);
E = zeros(1, nx);

rho_save = zeros(4,nx);
u_save = zeros(4,nx);
p_save = zeros(4,nx);
time = [0.0,0.05,0.1,0.2,0,3];
idx = 1;

for i = 1:nx
    if x(i) < 0.5
        rho(i) = rho_L;
        rho_u(i) = rho_L * u_L;
        E(i) = p_L / (gamma - 1) + 0.5 * rho_L * u_L^2;
    else
        rho(i) = rho_R;
        rho_u(i) = rho_R * u_R;
        E(i) = p_R / (gamma - 1) + 0.5 * rho_R * u_R^2;
    end

end
u = rho_u ./ rho;
p = (gamma - 1) * (E - 0.5 * rho .* u.^2);

rho_save(idx,:) = rho;
u_save(idx,:) = u;
p_save(idx,:) = p;
idx = idx + 1;

switch lower(scheme)
    case 'rusanov'
        % Rusanov scheme
        for n = 1:nt
            t = n * dt;
        
            % Create copies for updating
            rho_new = rho;
            rho_u_new = rho_u;
            E_new = E;
        
            for i = 2:nx-1
                % Primitive variables
                u_im1 = rho_u(i-1) / rho(i-1);
                u_i = rho_u(i) / rho(i);
                u_ip1 = rho_u(i+1) / rho(i+1);
        
                p_im1 = (gamma - 1) * (E(i-1) - 0.5 * rho(i-1) * u_im1^2);
                p_i = (gamma - 1) * (E(i) - 0.5 * rho(i) * u_i^2);
                p_ip1 = (gamma - 1) * (E(i+1) - 0.5 * rho(i+1) * u_ip1^2);
        
                a_im1 = sqrt(gamma * p_im1 / rho(i-1));
                a_i = sqrt(gamma * p_i / rho(i));
                a_ip1 = sqrt(gamma * p_ip1 / rho(i+1));
        
                % Fluxes
                F_im1 = [rho_u(i-1); 
                         rho_u(i-1) * u_im1 + p_im1; 
                         (E(i-1) + p_im1) * u_im1];
                F_i = [rho_u(i); 
                       rho_u(i) * u_i + p_i; 
                       (E(i) + p_i) * u_i];
                F_ip1 = [rho_u(i+1); 
                         rho_u(i+1) * u_ip1 + p_ip1; 
                         (E(i+1) + p_ip1) * u_ip1];
        
                lambda_im = max(abs(u_i) + a_i, abs(u_im1) + a_im1);
                lambda_ip = max(abs(u_i) + a_i, abs(u_ip1) + a_ip1);
        
                F_num_im = 0.5 * (F_im1 + F_i) - 0.5 * lambda_im * ([rho(i); rho_u(i); E(i)] - [rho(i-1); rho_u(i-1); E(i-1)]);
                F_num_ip = 0.5 * (F_i + F_ip1) - 0.5 * lambda_ip * ([rho(i+1); rho_u(i+1); E(i+1)] - [rho(i); rho_u(i); E(i)]);
        
                % Update
                rho_new(i) = rho(i) - dt/dx * (F_num_ip(1) - F_num_im(1));
                rho_u_new(i) = rho_u(i) - dt/dx * (F_num_ip(2) - F_num_im(2));
                E_new(i) = E(i) - dt/dx * (F_num_ip(3) - F_num_im(3));
            end
        
            % Boundary conditions
            rho_new(1) = rho_new(2);
            rho_u_new(1) = rho_u_new(2);
            E_new(1) = E_new(2);
            rho_new(nx) = rho_new(nx-1);
            rho_u_new(nx) = rho_u_new(nx-1);
            E_new(nx) = E_new(nx-1);
        
            % Update state
            rho = rho_new;
            rho_u = rho_u_new;
            E = E_new;
        
            % Compute primitive variables
            u = rho_u ./ rho;
            p = (gamma - 1) * (E - 0.5 * rho .* u.^2);
            

            % Save if close to a target time
            if abs(t - time(idx)) < dt/2
                rho_save(idx,:) = rho;
                u_save(idx,:) = u;
                p_save(idx,:) = p;
                idx = idx + 1;
            end
        end
        
    case 'jst'
        k_2 = 0.55; 
        k_4 = 1.5/128;
        
        for n = 1:nt
            t = dt *n;
            rho_new = rho;
            rho_u_new = rho_u;
            E_new = E;
            
            for i = 4:nx-3 
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

            % Compute primitive variables
            u = rho_u ./ rho;
            p = (gamma - 1) * (E - 0.5 * rho .* u.^2);
            

            % Save if close to a target time
            if abs(t - time(idx)) < dt/2
                rho_save(idx,:) = rho;
                u_save(idx,:) = u;
                p_save(idx,:) = p;
                idx = idx + 1;
            end
        end
    case 'roe'
    for n = 1:nt
        t = dt * n;
        rho_new = rho;
        rho_u_new = rho_u;
        E_new = E;
        
        for i = 2:nx-1
            % Left and right states for interface i-1/2
            rho_L = rho(i-1);
            rho_u_L = rho_u(i-1);
            E_L = E(i-1);
            
            rho_R = rho(i);
            rho_u_R = rho_u(i);
            E_R = E(i);
            
            % Compute primitive variables for left and right states
            u_L = rho_u_L / rho_L;
            u_R = rho_u_R / rho_R;
            
            p_L = (gamma - 1) * (E_L - 0.5 * rho_L * u_L^2);
            p_R = (gamma - 1) * (E_R - 0.5 * rho_R * u_R^2);
            
            H_L = (E_L + p_L) / rho_L;  % Left enthalpy
            H_R = (E_R + p_R) / rho_R;  % Right enthalpy
            
            % Compute Roe-averaged quantities
            sqrt_rho_L = sqrt(rho_L);
            sqrt_rho_R = sqrt(rho_R);
            
            inv_denom = 1 / (sqrt_rho_L + sqrt_rho_R);

            rho_roe = sqrt(rho_L* rho_R);
            
            % Roe-averaged velocity
            u_roe = (sqrt_rho_L * u_L + sqrt_rho_R * u_R) * inv_denom;
            
            % Roe-averaged enthalpy
            H_roe = (sqrt_rho_L * H_L + sqrt_rho_R * H_R) * inv_denom;
            
            % Roe-averaged sound speed
            a_roe = sqrt((gamma - 1) * (H_roe - 0.5 * u_roe^2));
            
            epsilon = a_roe*0.5;

            % Compute eigenvalues
            lambda_1 = abs(u_roe - a_roe);
            lambda_2 = abs(u_roe);
            lambda_3 = abs(u_roe + a_roe);
            
            % Compute differences in conservative variables
            drho = rho_R - rho_L;
            drho_u = rho_u_R - rho_u_L;
            dE = E_R - E_L;
            
            % Compute differences in primitive variables for wave strength calculation
            dp = p_R - p_L;
            du = u_R - u_L;
            
            % Calculate wave strengths (alpha)
            alpha_1 = (1/(2*a_roe^2)) * (dp - rho_roe*a_roe*du);
            alpha_2 = drho - (dp/(a_roe^2));
            alpha_3 = (1/(2*a_roe^2)) * (dp + rho_roe*a_roe*du);
            
            % Compute the right eigenvectors
            r_1 = [1; u_roe - a_roe; H_roe - u_roe*a_roe];
            r_2 = [1; u_roe; 0.5*u_roe^2];
            r_3 = [1; u_roe + a_roe; H_roe + u_roe*a_roe];
            
            % Compute the numerical flux using Roe's approximate Riemann solver
            F_L = [rho_u_L; rho_u_L * u_L + p_L; (E_L + p_L) * u_L];
            F_R = [rho_u_R; rho_u_R * u_R + p_R; (E_R + p_R) * u_R];
            
            % Numerical flux for i-1/2 interface
            F_im = 0.5 * (F_L + F_R) - 0.5 * (lambda_1 * alpha_1 * r_1 + lambda_2 * alpha_2 * r_2 + lambda_3 * alpha_3 * r_3);
            
            % Right and left states for interface i+1/2
            rho_L = rho(i);
            rho_u_L = rho_u(i);
            E_L = E(i);
            
            rho_R = rho(i+1);
            rho_u_R = rho_u(i+1);
            E_R = E(i+1);
            
            % Compute primitive variables
            u_L = rho_u_L / rho_L;
            u_R = rho_u_R / rho_R;
            
            p_L = (gamma - 1) * (E_L - 0.5 * rho_L * u_L^2);
            p_R = (gamma - 1) * (E_R - 0.5 * rho_R * u_R^2);
            
            H_L = (E_L + p_L) / rho_L;
            H_R = (E_R + p_R) / rho_R;
            
            % Compute Roe-averaged quantities
            sqrt_rho_L = sqrt(rho_L);
            sqrt_rho_R = sqrt(rho_R);
            
            inv_denom = 1 / (sqrt_rho_L + sqrt_rho_R);
            
            % Roe-averaged values
            u_roe = (sqrt_rho_L * u_L + sqrt_rho_R * u_R) * inv_denom;
            H_roe = (sqrt_rho_L * H_L + sqrt_rho_R * H_R) * inv_denom;
            a_roe = sqrt((gamma - 1) * (H_roe - 0.5 * u_roe^2));
            rho_roe = sqrt(rho_L * rho_R);
            
            % Compute eigenvalues
            lambda_1 = abs(u_roe - a_roe);
            if lambda_1 <epsilon
                lambda_1 = (lambda_1*lambda_1 + epsilon*epsilon)/(2*epsilon);
            end
            lambda_2 = abs(u_roe);
            if lambda_1 <epsilon
                lambda_1 = (lambda_2*lambda_2 + epsilon*epsilon)/(2*epsilon);
            end
            lambda_3 = abs(u_roe + a_roe);
            if lambda_3 <epsilon
                lambda_3 = (lambda_3*lambda_3 + epsilon*epsilon)/(2*epsilon);
            end
            
            % Compute differences
            drho = rho_R - rho_L;
            drho_u = rho_u_R - rho_u_L;
            dE = E_R - E_L;
            
            dp = p_R - p_L;
            du = u_R - u_L;
            
            % Calculate wave strengths
            alpha_1 = (1/(2*a_roe^2)) * (dp - rho_roe*a_roe*du);
            alpha_2 = drho - (dp/(a_roe^2));
            alpha_3 = (1/(2*a_roe^2)) * (dp + rho_roe*a_roe*du);
            
            % Compute right eigenvectors
            r_1 = [1; u_roe - a_roe; H_roe - u_roe*a_roe];
            r_2 = [1; u_roe; 0.5*u_roe^2];
            r_3 = [1; u_roe + a_roe; H_roe + u_roe*a_roe];
            
            % Compute fluxes
            F_L = [rho_u_L; rho_u_L * u_L + p_L; (E_L + p_L) * u_L];
            F_R = [rho_u_R; rho_u_R * u_R + p_R; (E_R + p_R) * u_R];
            
            % Numerical flux for i+1/2 interface
            F_ip = 0.5 * (F_L + F_R) - 0.5 * (lambda_1 * alpha_1 * r_1 + lambda_2 * alpha_2 * r_2 + lambda_3 * alpha_3 * r_3);
            
            % Update solution
            rho_new(i) = rho(i) - dt/dx * (F_ip(1) - F_im(1));
            rho_u_new(i) = rho_u(i) - dt/dx * (F_ip(2) - F_im(2));
            E_new(i) = E(i) - dt/dx * (F_ip(3) - F_im(3));
        end
        
        % Update boundary conditions (simple extrapolation)
        for i = 1:1
            rho_new(i) = rho_new(2);
            rho_u_new(i) = rho_u_new(2);
            E_new(i) = E_new(2);
            
            rho_new(nx-i+1) = rho_new(nx-1);
            rho_u_new(nx-i+1) = rho_u_new(nx-1);
            E_new(nx-i+1) = E_new(nx-1);
        end
        
        % Update values for next time step
        rho = rho_new;
        rho_u = rho_u_new;
        E = E_new;
        
        % Compute primitive variables
        u = rho_u ./ rho;
        p = (gamma - 1) * (E - 0.5 * rho .* u.^2);
        
        % Save if close to a target time
        if abs(t - time(idx)) < dt/2
            rho_save(idx,:) = rho;
            u_save(idx,:) = u;
            p_save(idx,:) = p;
            idx = idx + 1;
        end
    end

    otherwise
        error('Unknown scheme: %s. Available options: rusanov, jst', scheme);
       
end

% Create unique filename based on grid and time step
file = "roe" + string(nx) + "_new_" + string(dt);

% Time labels for legend
time_labels = ["t=0.0", "t=0.05", "t=0.1", "t=0.25"];

% === Figure 1: Density ===
figure('Position', [100, 100, 800, 300]);
hold on;
colors = lines(4); % 4 distinguishable colors
for i = 1:4
    plot(x, rho_save(i,:), 'LineWidth', 2, 'Color', colors(i,:));
end
xlabel('Position (x)');
ylabel('\rho');
title('Density at Different Times');
legend(time_labels, 'Location', 'best');
grid on;
hold off;
saveas(gcf, file + "_density.png");

% === Figure 2: Velocity ===
figure('Position', [100, 550, 800, 300]);
hold on;
for i = 1:4
    plot(x, u_save(i,:), 'LineWidth', 2, 'Color', colors(i,:));
end
xlabel('Position (x)');
ylabel('u');
title('Velocity at Different Times');
legend(time_labels, 'Location', 'best');
grid on;
hold off;
saveas(gcf, file + "_velocity.png");

% === Figure 3: Pressure ===
figure('Position', [750, 100, 800, 300]);
hold on;
for i = 1:4
    plot(x, p_save(i,:), 'LineWidth', 2, 'Color', colors(i,:));
end
xlabel('Position (x)');
ylabel('p');
title('Pressure at Different Times');
legend(time_labels, 'Location', 'best');
grid on;
hold off;
saveas(gcf, file + "_pressure.png");
