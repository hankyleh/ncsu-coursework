%
% This script sets up the problem data 
% for solution of the inverse heat equation.
%

%
% The true initial state, which we want to reconstruct
%
u0  = @(x,y)( ...
              ((x-.125).^2 + (y-.125).^2 <= .01) * 10 + ...
              ((x-.5).^2 + (y-.5).^2 <= .01) * 5);

kappa = .0025;      % diffusion coefficient

%
% Spatial discretization
% 
nx = 2^8;
h = 1/nx;
xi = [0 : h : 1]'; yi = xi;
[Xi Yi] = meshgrid(xi,xi);

% the spatial discretization matrix 
h = 1/nx;
m = nx-1;
I = speye(m);
e = ones(m,1);
T = spdiags([-e 4*e -e],[-1 0 1],m,m);
S = spdiags([-e -e],[-1 1],m,m);
K = kron(I,T) + kron(S,I);
A = kappa * (1/h^2) * K; 

%
% Time discretization
%
t0 = 0;
tf = 8;
nt = 300 + 1;
dt = tf / (nt-1); 
ti = [t0 : dt : tf];


%
% record model data required by the forward solver
%
model_data.A = A;
model_data.h = h;
model_data.nx = nx;
model_data.m = m;
model_data.tf = tf;
model_data.nt = nt;
model_data.dt = dt;
model_data.R = (1/h^2)*K;

%
% Synthesize observational data
%
ut = u0(Xi,Yi);
ut = ut(2:end-1,2:end-1);
ut = ut(:);
ud0 = heat_forward_solve_2d(ut, model_data);
s = norm(ud0, 'inf')*.05;
ud = ud0 + s * randn((nx-1)^2,1);


F = @(v)(heat_forward_solve_2d(v, model_data));
A = @(v)(F(F(v)));