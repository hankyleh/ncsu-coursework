runtimes = [0 0 0];

tic
[U_array ti Xi Yi] = solve_heat_2d;
fprintf("Runtime %.1f seconds"+newline, toc);
runtimes(1) = toc;

tic
[U_array ti Xi Yi] = chol_solve;
fprintf("Runtime %.1f seconds"+newline, toc);
runtimes(2) = toc;

%%
tic
[U_array ti Xi Yi] = chol_symamd_solve;
fprintf("Runtime %.1f seconds"+newline, toc);
runtimes(3) = toc;



%%
maxZ = max(U_array{1}(:))
for t = interp1(ti, 1:length(ti), [0, 1, 2, 4, 7, 10], "nearest")
    figure()
    surf(Xi, Yi, U_array{t})
    axis square tight
    zlim([0 maxZ]);
    shading interp
    box on;

    set(gca, 'fontsize', 20);
    xlabel('x');
    ylabel('y');
    zlabel('u');
    title(sprintf("solution at %.1f", ti(t)));
    view(-10,30);
    saveas(gcf, sprintf("heat_eqn_t%i.png", round(ti(t))))
end


