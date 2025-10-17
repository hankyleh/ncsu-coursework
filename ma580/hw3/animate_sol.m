function animate_sol(ti, Xi, Yi, U_array)
%
% animates the solution to the heat equation 
%
% the input arguments to this function are
% the outputs from the heat equation solver.

maxZ = max(U_array{1}(:));
close all;
nt = length(ti);
for i = 1 : nt
   surf(Xi, Yi, U_array{i});

   axis square tight
   zlim([0 maxZ]);
   shading interp
   box on;

   set(gca, 'fontsize', 20);
   xlabel('x');
   ylabel('y');
   zlabel('u');
   title(['solution at t = ' num2str(ti(i), '%4.2f')]);
   view(-10,30);

   pause(0.0001);
end