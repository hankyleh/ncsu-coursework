% HW 6, problem 1.
% nonlinear PDE, -kLu + cu^3 = f
    % L = laplacian
    % k, c = constants
    % f = -2pi[cos(2 pi x_1)sin^2(pi x_2) + sin^2(pi x_1) cos(2 pi x_2)]

% Solves using Newton Iteration

close all

% physical constants
kap = 0.1;
c = 100;

% iteration controls
trel = 1e-10;
tabs = 1e-10;
k_max = 20

iters = [];
end_r = [];

% m = log_2(n_x)
% perform iteration and record results for each case
for m = 3:8
    n = 2^m; % number of points
    
    % problem domain
    x1 = linspace(0, 1, n);
    x2 = linspace(0, 1, n);
    [X1,  X2] = meshgrid(x1, x2);
    
    % width between points
    h = 1/n; 
    
    % discretize differential operator, 1d
    v = ones(n, 1)/h^2;
    lapl_1d = spdiags([v -2*v v], -1:1, n, n);
    I = speye(n);
    %discrete laplacian in 2d
    lapl_2d = kron(lapl_1d, I) + kron(I, lapl_1d); 
    A = -kap*lapl_2d;
    
    % initial guess, zeros
    v0 = zeros(n^2, 1);
    v_iter = v0;
    r = norm(func_eval(A, v0, c, X1, X2, n), 2);
    r0 = r;
    
    residuals = r;
    tol = trel*r + tabs;
    
    
    F  = func_eval(A, v_iter, c, X1, X2, n);
    k = 1;
    
    %figure()
    while r > tol && k < k_max
        k = k + 1;
        J = jacobian(A, c, v_iter, n);
        
        s = -J\F;
        v_iter = v_iter + s;
        F  = func_eval(A, v_iter, c, X1, X2, n);
    
        r = norm(F, 2);
        residuals = [residuals r/r0];
        %surf(x1, x2, reshape(v_prev, n, n), LineStyle="none")
        %zlim(gca, [-1, 3.5])
        %pause(0.06)
    end
    end_r = [end_r r/r0];
    iters = [iters k];
end

%% 
figure()

yyaxis("left")
scatter(0:length(residuals)-1, residuals, "filled")
hold on
scatter(1:length(residuals)-1, residuals(1:end-1).^2, "blue")
set(gca, 'YScale', 'log')
ylabel("$$|r_k|$$", "interpreter", "latex", FontSize=13.5)

yyaxis("right")
scatter(1:length(residuals)-1,residuals(2:end)./(residuals(1:end-1).^2))
set(gca, "YScale", "log")
ylabel("$$||r_k|| / ||r_{k-1}||^2$$", ...
    Interpreter="latex", FontSize=13.5)
legend("$$r_k$$", "$$r_{k-1}^2$$", "$$\hat{c}$$", ...
    "interpreter", "latex", fontsize=11)

title("Relative residuals")
xlabel("Iteration index")
saveas(gcf, sprintf("c%i_res_m%i.png", c, m))

function phi = nonlin(U)
    phi = U.^3;
end

function dphi = dnonlin(U)
    dphi = 3*U.^2;
end

function f = src(x1, x2, n)
    f = -2*pi^2*(cos(2*pi*x1).*sin(pi*x2).^2 ...
        + sin(pi*x1).^2.*cos(2*pi*x2));
    f = reshape(f, n^2, 1);
end

function F = func_eval(A, u, c, X1, X2, n)
    F = (A*u) + (c* nonlin(u)) - src(X1, X2, n);
end

function J = jacobian(A, c, u, n)
    J = A + c*spdiags(dnonlin(u), 0, n^2, n^2); 
end