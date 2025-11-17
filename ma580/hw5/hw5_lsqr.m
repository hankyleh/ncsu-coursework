clear all
init_heat_2d

alpha = 1e-2;

I = speye((nx-1)^2);
A = (I + tf* model_data.A);

R = speye(size(model_data.A));
%R = model_data.R;

% M = (A' * A)+ (sqrt(alpha)*R);

M = @(x)(forward_heat_solve_2d(x, model_data) + alpha*R*x);


b = heat_forward_solve_2d(ud, model_data);
U = zeros(size(b));
U(:, 1) = pcg(@regularized, b, 1e-6, 20);

% for n = 1 : nt
%     b = heat_forward_solve_2d(U(:, n), model_data);
%     U(:, n+1) = pcg(M'*M, b, 1e-6, 100);
% end


%% reshape

z = reshape(U(:, end), [nx-1, nx-1]);


%% plots
close all

figure()
surf(Xi, Yi, u0(Xi, Yi), LineStyle="none")
view(0, 90)
colorbar

figure()
surf(xi(2:end-1), yi(2:end-1), reshape(ud, [nx-1, nx-1]), LineStyle="none")
view(0, 90)
colorbar

figure()
surf(xi(2:end-1), yi(2:end-1), z, LineStyle="none")
view(0, 90)
colorbar