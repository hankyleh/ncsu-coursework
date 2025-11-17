function b = regularized(x)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

alpha = 1e-8
x = speye(size(model_data.A))
b = heat_forward_solve_2d(x, model_data) + alpha*R*x