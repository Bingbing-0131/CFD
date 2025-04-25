gamma = 1.4; 
rho_L = 1.0; u_L = 0.0; p_L = 1.0;   
rho_R = 0.125; u_R = 0.0; p_R = 0.1;

x_min = 0; x_max = 1.0;
t_final = 0.2;
epsilon = 0.1;  

nx = 200;
dx = (x_max - x_min) / nx;

dt = 1e-3;
nt = floor(t_final / dt) + 1;
dt = t_final / nt; 

x = linspace(x_min + dx/2, x_max - dx/2, nx);  
rho = zeros(1, nx);
rho_u = zeros(1, nx);
E = zeros(1, nx);


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

rho_init = rho;
rho_u_init = rho_u;
E_init = E;


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
    end
        rho_save_roe = rho;
        u_save_roe = rho_u ./ rho;
        p_save_roe = (gamma - 1) * (E - 0.5 * rho .* (u_save_roe).^2);


        rho = rho_init;
        rho_u = rho_u_init;
        E = E_init;
     
        % Time integration loop
        for n = 1:nt
            % Store original values for RK stages
            rho_0 = rho;
            rho_u_0 = rho_u;
            E_0 = E;
            t = n * dt;
            
            % First RK stage
            [drhodt, drhoudt, dEdt] = compute_residual(rho_0, rho_u_0, E_0, nx, dx, gamma, use_entropy_fix);
            rho_1 = rho_0 + dt * drhodt;
            rho_u_1 = rho_u_0 + dt * drhoudt;
            E_1 = E_0 + dt * dEdt;
            
            % Apply boundary conditions
            rho_1(1) = rho_1(2);
            rho_u_1(1) = rho_u_1(2);
            E_1(1) = E_1(2);
            rho_1(nx) = rho_1(nx-1);
            rho_u_1(nx) = rho_u_1(nx-1);
            E_1(nx) = E_1(nx-1);
            
            % Second RK stage
            [drhodt, drhoudt, dEdt] = compute_residual(rho_1, rho_u_1, E_1, nx, dx, gamma, use_entropy_fix);
            rho_2 = 0.75 * rho_0 + 0.25 * rho_1 + 0.25 * dt * drhodt;
            rho_u_2 = 0.75 * rho_u_0 + 0.25 * rho_u_1 + 0.25 * dt * drhoudt;
            E_2 = 0.75 * E_0 + 0.25 * E_1 + 0.25 * dt * dEdt;
            
            % Apply boundary conditions
            rho_2(1) = rho_2(2);
            rho_u_2(1) = rho_u_2(2);
            E_2(1) = E_2(2);
            rho_2(nx) = rho_2(nx-1);
            rho_u_2(nx) = rho_u_2(nx-1);
            E_2(nx) = E_2(nx-1);
            
            % Third RK stage
            [drhodt, drhoudt, dEdt] = compute_residual(rho_2, rho_u_2, E_2, nx, dx, gamma, use_entropy_fix);
            rho = (1/3) * rho_0 + (2/3) * rho_2 + (2/3) * dt * drhodt;
            rho_u = (1/3) * rho_u_0 + (2/3) * rho_u_2 + (2/3) * dt * drhoudt;
            E = (1/3) * E_0 + (2/3) * E_2 + (2/3) * dt * dEdt;
            
            % Apply boundary conditions
            rho(1) = rho(2);
            rho_u(1) = rho_u(2);
            E(1) = E(2);
            rho(nx) = rho(nx-1);
            rho_u(nx) = rho_u(nx-1);
            E(nx) = E(nx-1);
        end

        rho_save_tvd = rho;
u_save_tvd = rho_u ./ rho;
p_save_tvd = (gamma - 1) * (E - 0.5 * rho .* (u_save_tvd).^2);


% === Comparison Plot at t = 0.2 ===
figure('Position', [100, 100, 800, 300]);
plot(x, rho_save_roe, 'b-', 'LineWidth', 2); hold on;
plot(x, rho_save_tvd, 'r--', 'LineWidth', 2);
xlabel('Position (x)');
ylabel('\rho');
title('Density Comparison at t = 0.2');
legend('Roe', 'TVD');
grid on;
saveas(gcf, "compare_density_t02.png");

figure('Position', [100, 450, 800, 300]);
plot(x, u_save_roe, 'b-', 'LineWidth', 2); hold on;
plot(x, u_save_tvd, 'r--', 'LineWidth', 2);
xlabel('Position (x)');
ylabel('u');
title('Velocity Comparison at t = 0.2');
legend('Roe', 'TVD');
grid on;
saveas(gcf, "compare_velocity_t02.png");

figure('Position', [100, 800, 800, 300]);
plot(x, p_save_roe, 'b-', 'LineWidth', 2); hold on;
plot(x, p_save_tvd, 'r--', 'LineWidth', 2);
xlabel('Position (x)');
ylabel('p');
title('Pressure Comparison at t = 0.2');
legend('Roe', 'TVD');
grid on;
saveas(gcf, "compare_pressure_t02.png");


function [drhodt, drhoudt, dEdt] = compute_residual(rho, rho_u, E, nx, dx, gamma, use_entropy_fix)
    % Initialize residual arrays
    drhodt = zeros(size(rho));
    drhoudt = zeros(size(rho_u));
    dEdt = zeros(size(E));
    
    % Initialize interface fluxes arrays
    F_im = zeros(nx, 3);
    F_ip = zeros(nx, 3);
    
    % Loop through interior cells to compute fluxes
    for i = 2:nx-1
        for offset = 0:1
            il = i + offset - 1;  % Left cell index
            ir = i + offset;      % Right cell index
            
            % Get cell states
            rho_L = rho(il);
            rho_u_L = rho_u(il);
            E_L = E(il);
            rho_R = rho(ir);
            rho_u_R = rho_u(ir);
            E_R = E(ir);
            
            % Compute primitive variables
            u_L = rho_u_L / rho_L;
            u_R = rho_u_R / rho_R;
            p_L = (gamma - 1) * (E_L - 0.5 * rho_L * u_L^2);
            p_R = (gamma - 1) * (E_R - 0.5 * rho_R * u_R^2);
            H_L = (E_L + p_L) / rho_L;  % Enthalpy
            H_R = (E_R + p_R) / rho_R;
            
            % Roe averages for linearization
            sqrt_rho_L = sqrt(rho_L);
            sqrt_rho_R = sqrt(rho_R);
            denom_inv = 1 / (sqrt_rho_L + sqrt_rho_R);
            
            rho_roe = sqrt(rho_L * rho_R);
            u_roe = (sqrt_rho_L * u_L + sqrt_rho_R * u_R) * denom_inv;
            H_roe = (sqrt_rho_L * H_L + sqrt_rho_R * H_R) * denom_inv;
            a_roe = sqrt((gamma - 1) * (H_roe - 0.5 * u_roe^2));
            
            % Left and right eigenvectors at the Roe state
            % Left eigenvectors (rows of L)
            beta1 = 0.5/(a_roe^2);
            beta2 = 1/(a_roe);
            
            L = [
                [(gamma-1)*u_roe*u_roe*beta1+u_roe*beta2, -(gamma-1)*u_roe*beta1-beta2, (gamma-1)*beta1];
                [1-gamma*u_roe*u_roe*beta1, gamma*u_roe*beta1, -gamma*beta1];
                [(gamma-1)*u_roe*u_roe*beta1-u_roe*beta2, -(gamma-1)*u_roe*beta1+beta2, (gamma-1)*beta1]
            ];
            
            % Right eigenvectors (columns of R)
            R = [
                [1, 1, 1];
                [u_roe-a_roe, u_roe, u_roe+a_roe];
                [H_roe-u_roe*a_roe, 0.5*u_roe^2, H_roe+u_roe*a_roe]
            ];
            
            % Apply TVD reconstruction with characteristic variables
            if (il > 1 && ir < nx)
                % Get neighboring cell indices
                ill = il - 1;  % Left-left cell
                irr = ir + 1;  % Right-right cell
                
                % Create conservative variable vectors
                U_ill = [rho(ill); rho_u(ill); E(ill)];
                U_il = [rho(il); rho_u(il); E(il)];
                U_ir = [rho(ir); rho_u(ir); E(ir)];
                U_irr = [rho(irr); rho_u(irr); E(irr)];
                
                % Compute differences in conservative variables
                dU_L = U_il - U_ill;
                dU_C = U_ir - U_il;
                dU_R = U_irr - U_ir;
                
                % Transform to characteristic variables
                W_L = L * dU_L;
                W_C = L * dU_C;
                W_R = L * dU_R;
                
                % Apply minmod limiter to characteristic variables
                W_limL = zeros(3, 1);
                W_limR = zeros(3, 1);
                
                for k = 1:3
                    % Left state reconstruction
                    W_limL(k) = minmod(W_L(k), W_C(k));
                    
                    % Right state reconstruction
                    W_limR(k) = minmod(W_C(k), W_R(k));
                end
                
                % Transform back to conservative variables
                dU_limL = R * W_limL;
                dU_limR = R * W_limR;
                
                % Update left and right states at interface
                U_L = U_il + 0.5 * dU_limL;
                U_R = U_ir - 0.5 * dU_limR;
                
                % Extract updated conservative variables
                rho_L = U_L(1);
                rho_u_L = U_L(2);
                E_L = U_L(3);
                
                rho_R = U_R(1);
                rho_u_R = U_R(2);
                E_R = U_R(3);
                
                % Recompute primitive variables
                u_L = rho_u_L / rho_L;
                u_R = rho_u_R / rho_R;
                p_L = max(1e-10, (gamma - 1) * (E_L - 0.5 * rho_L * u_L^2));
                p_R = max(1e-10, (gamma - 1) * (E_R - 0.5 * rho_R * u_R^2));
                H_L = (E_L + p_L) / rho_L;
                H_R = (E_R + p_R) / rho_R;
            end
            
            % Recompute Roe averages with reconstructed states if needed
            sqrt_rho_L = sqrt(rho_L);
            sqrt_rho_R = sqrt(rho_R);
            denom_inv = 1 / (sqrt_rho_L + sqrt_rho_R);
            
            rho_roe = sqrt(rho_L * rho_R);
            u_roe = (sqrt_rho_L * u_L + sqrt_rho_R * u_R) * denom_inv;
            H_roe = (sqrt_rho_L * H_L + sqrt_rho_R * H_R) * denom_inv;
            a_roe = sqrt((gamma - 1) * (H_roe - 0.5 * u_roe^2));
            
            % Eigenvalues
            lambda_1 = abs(u_roe - a_roe);
            lambda_2 = abs(u_roe);
            lambda_3 = abs(u_roe + a_roe);
            
            % Apply entropy fix
            if use_entropy_fix
                epsilon = 0.1 * a_roe;
                if lambda_1 < epsilon
                    lambda_1 = (lambda_1^2 + epsilon^2) / (2*epsilon);
                end
                if lambda_2 < epsilon
                    lambda_2 = (lambda_2^2 + epsilon^2) / (2*epsilon);
                end
                if lambda_3 < epsilon
                    lambda_3 = (lambda_3^2 + epsilon^2) / (2*epsilon);
                end
            end
            
            % Wave strengths
            dp = p_R - p_L;
            du = u_R - u_L;
            drho = rho_R - rho_L;
            
            alpha_1 = (1/(2*a_roe^2)) * (dp - rho_roe * a_roe * du);
            alpha_2 = drho - dp / a_roe^2;
            alpha_3 = (1/(2*a_roe^2)) * (dp + rho_roe * a_roe * du);
            
            % Right eigenvectors
            r_1 = [1; u_roe - a_roe; H_roe - u_roe * a_roe];
            r_2 = [1; u_roe; 0.5 * u_roe^2];
            r_3 = [1; u_roe + a_roe; H_roe + u_roe * a_roe];
            
            % Physical fluxes
            F_L = [rho_u_L; rho_u_L * u_L + p_L; (E_L + p_L) * u_L];
            F_R = [rho_u_R; rho_u_R * u_R + p_R; (E_R + p_R) * u_R];
            
            % Roe flux
            flux = 0.5 * (F_L + F_R) - 0.5 * (lambda_1 * alpha_1 * r_1 + lambda_2 * alpha_2 * r_2 + lambda_3 * alpha_3 * r_3);
            
            % Store flux based on offset
            if offset == 0
                F_im(i, :) = flux';
            else
                F_ip(i, :) = flux';
            end
        end
    end
    
    % Compute residuals for interior cells
    for i = 2:nx-1
        drhodt(i) = -(F_ip(i, 1) - F_im(i, 1)) / dx;
        drhoudt(i) = -(F_ip(i, 2) - F_im(i, 2)) / dx;
        dEdt(i) = -(F_ip(i, 3) - F_im(i, 3)) / dx;
    end
end

% Minmod slope limiter
function result = minmod(a, b)
    if a * b <= 0
        result = 0;
    elseif abs(a) < abs(b)
        result = a;
    else
        result = b;
    end
end