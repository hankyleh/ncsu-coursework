function out = heat_forward_solve_2d(v, model_data)
% solves the following PDE using second order accurate 
% finite-difference approaximation in space and implicit Euler
% in time:
%
%   u_t + Laplacian u = 0         in Omega x [0, T]
%   u(x, 0)           = u0(x)     in Omega
%   
%   with homogeneous Dirichlet boundary condition.
%  
%   output: 
%      out:    u at the final time

%
% PDE parameters
%
A  = model_data.A;
%x  = model_data.x;
h  = model_data.h;
nx = model_data.nx;
m  = model_data.m; 
tf = model_data.tf;
nt = model_data.nt;
dt = model_data.dt;
R  = model_data.R;


%
% time-stepping
%
U = zeros((nx-1)^2,nt+1);
U(:,1) = v(:);
I = speye((nx-1)^2);

tic 
L = (I + dt * A);
p = symamd(L);
R = chol(L(p, p));
Rt = R';
for n = 1 : nt
   U(p,n+1) = R \ (Rt \ U(p,n));
end
t_solve = toc;

out = U(:,end); 
%disp('heat solve');
