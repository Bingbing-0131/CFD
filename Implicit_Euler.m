convection_equation(100, 0.5, [0.0,0.1,1.0,10.0]);

function convection_equation(Mx, c, t_vals)
    % Inputs:
    %   Mx   - Number of spatial grid points
    %   c - CFL number, c = a * Delta t / Delta x
    %   t_vals  -  n time points, from small to large
    %   method - 'upwind' or 'Euler implicit'

    % Basic parameters
    a = 1;
    x_min = -0.5; x_max = 0.5;
    dx = (x_max - x_min) / Mx;
    dt = c * dx / a;
    x = linspace(x_min, x_max, Mx+1);

    % Initial condition
    u = transpose(zeros(size(x)));
    u(x > -0.25 & x < 0.25) = 1;
   % parameter matrix of implicit format
     A = eye(Mx+1) + c/2 * diag(ones(Mx,1),1) - c/2 * diag(ones(Mx,1),-1)
     A(1,2) = c/2; A(1,end)=-c/2;
     A(end,1) = c/2; A(end,end-1) = -c/2;
            
     figure; hold on;
            legends = {};

            for i = 1:length(t_vals)
                t_target = t_vals(i);
                t = 0;
                
                % time advancement
                while t < t_target
                    u = A \ u;
                    t = t + dt;
                end
                
                plot(x, u);
                hold on
                legends{end+1} = [sprintf('t = %.1f', t_target)];

            end
            title("Implicit Euler")
            xlabel('$x$',Interpreter='latex');
            ylabel('$u(x,t)$',Interpreter='latex');
            legend(legends,Location='northwest');
            grid on;
            hold off
        
    end
