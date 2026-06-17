function plot_density_diffusion(file_name, output_dir, plot_kind, plane)
% Plot BioFVM `multiscale_microenvironment` MATLAB output on a middle slice.
%
% Usage:
%   plot_density_diffusion('output/density_diffusion_final.mat', 'output/plots', 'contour', 'xy')
%   plot_density_diffusion('output/density_diffusion_final.mat', 'output/plots', 'surface', 'xy')

if nargin < 2 || isempty(output_dir)
    output_dir = '.';
end
if nargin < 3 || isempty(plot_kind)
    plot_kind = 'contour';
end
if nargin < 4 || isempty(plane)
    plane = 'xy';
end

if exist(output_dir, 'dir') ~= 7
    mkdir(output_dir);
end

loaded = load(file_name);
m = loaded.multiscale_microenvironment;

x = 1; y = 2; z = 3;
labels = {'x', 'y', 'z'};

switch lower(plane)
    case 'xy'
        needed_plane = [x, y];
    case 'xz'
        needed_plane = [x, z];
    case 'yz'
        needed_plane = [y, z];
    otherwise
        error('Unknown plane "%s". Use xy, xz, or yz.', plane);
end

cross_section_index = setdiff([1, 2, 3], needed_plane);
slice_values = unique(sort(m(cross_section_index, :)));
slice_index = max(1, floor((length(slice_values) + 1) / 2));
slice_value = slice_values(slice_index);
m = m(:, m(cross_section_index, :) == slice_value);

axis_a = unique(sort(m(needed_plane(1), :)));
axis_b = unique(sort(m(needed_plane(2), :)));

if length(axis_a) < 2 || length(axis_b) < 2
    error('Need at least two points in each plotted direction.');
end

step_a = abs(axis_a(2) - axis_a(1));
step_b = abs(axis_b(2) - axis_b(1));
min_a = axis_a(1) - step_a / 2;
max_a = axis_a(end) + step_a / 2;
min_b = axis_b(1) - step_b / 2;
max_b = axis_b(end) + step_b / 2;

num_a = length(axis_a);
num_b = length(axis_b);
scaled_a = 1 + floor(num_a * ((m(needed_plane(1), :) - min_a) / (max_a - min_a)));
scaled_b = 1 + floor(num_b * ((m(needed_plane(2), :) - min_b) / (max_b - min_b)));

[~, base_name, ~] = fileparts(file_name);

for density_row = 5:size(m, 1)
    density_index = density_row - 4;
    full_matrix = full(sparse(scaled_b, scaled_a, m(density_row, :), num_b, num_a));

    fig = figure('visible', 'off');
    if strcmpi(plot_kind, 'surface')
        surf(axis_a, axis_b, full_matrix);
        shading interp;
        zlabel('concentration');
    else
        contourf(axis_a, axis_b, full_matrix, 30, 'linecolor', 'none');
        axis image;
    end

    colorbar('FontSize', 12);
    title(sprintf('%s density %d %s slice at %s = %.6g', ...
        strrep(base_name, '_', '\_'), density_index, upper(plane), labels{cross_section_index}, slice_value), ...
        'FontSize', 12);
    xlabel(sprintf('%s (um)', labels{needed_plane(1)}));
    ylabel(sprintf('%s (um)', labels{needed_plane(2)}));

    output_file = fullfile(output_dir, sprintf('%s_density_%d_%s_%s.png', ...
        base_name, density_index, lower(plane), lower(plot_kind)));
    print(fig, output_file, '-dpng', '-r200');
    close(fig);
    fprintf('Wrote %s\n', output_file);
end
end
