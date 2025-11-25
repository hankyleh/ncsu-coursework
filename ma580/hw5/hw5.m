clear all
close all

global R alpha model_data
init_heat_2d()

%%
nt = model_data.nt;
nx = model_data.nx;

s = length(model_data.A);
q = sqrt(s);

R = model_data.A;
alpha = 10^(-1.75);
tol = 1e-8;

start = tic;

iter = zeros(1, nt+1);
flag = 0;
relres = iter;

U = zeros((nx-1)^2,nt+1);
U(:, 1) = ud;

model_data.nt = 1;


for n = 1:nt
    b = heat_forward_solve_2d(U(:, n), model_data);
    [U(:, n+1), flag, relres(n+1), iter(n+1)] = ...
        pcg(@normal_forward_heat, b, tol, 40,[],[],U(:, n));
    fprintf("time step %i \n", n)
end

fprintf("%.2f second runtime", toc-start)

%% plots
close all

figure()
plot(iter, LineWidth=2)
xlabel("Backward time step index")
ylabel("Iterations")
title("convergence")

figure()
plot(relres, LineWidth=2)
xlabel("Backward time step index")
ylabel("Relative Residual")


%%

figure()
s = surf(xi(2:end-1), yi(2:end-1), reshape(ud, [q, q]));
s.EdgeColor = 'none';
view(0,90)
colorbar

figure()
s = surf(xi(2:end-1), yi(2:end-1), reshape(ut, [q, q]));
s.EdgeColor = 'none';
view(0,90)
colorbar

for t = [300]
    figure()
    s = surf(xi(2:end-1), yi(2:end-1), reshape(U(:, t), [q, q]));
    s.EdgeColor = 'none';
    view(0,90)
    colorbar
    title(sprintf("%i", t))
end








function M = normal_forward_heat(v)
    global R alpha model_data
    
    M = heat_forward_solve_2d(...
        heat_forward_solve_2d(v, model_data), model_data) ...
        + (alpha*R*v);
end


