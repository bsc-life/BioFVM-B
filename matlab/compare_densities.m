% Compare BioFVM multiscale_microenvironment density outputs.
%
% Usage with Octave:
%   octave matlab/compare_densities.m ref.mat new.mat
%
% Usage from MATLAB/Octave workspace:
%   file_ref = '/tmp/biofvm_complete_test/density_diffusion_final.mat';
%   file_new = '/tmp/biofvmb_complete_test_np2/density_diffusion_final.mat';
%   compare_densities

warning('off', 'all');

args = {};
if exist('OCTAVE_VERSION', 'builtin')
    args = argv();
end

if numel(args) >= 2
    file_ref = args{1};
    file_new = args{2};
else
    if ~exist('file_ref', 'var') || isempty(file_ref)
        file_ref = '/tmp/biofvm_complete_test/density_diffusion_final.mat';
    end
    if ~exist('file_new', 'var') || isempty(file_new)
        file_new = '/tmp/biofvmb_complete_test_np2/density_diffusion_final.mat';
    end
end

if ~exist('equivalence_tolerance', 'var') || isempty(equivalence_tolerance)
    equivalence_tolerance = 1e-12;
end

fprintf('Reference file: %s\n', file_ref);
fprintf('New file:       %s\n', file_new);

ref_loaded = load(file_ref);
new_loaded = load(file_new);

if ~isfield(ref_loaded, 'multiscale_microenvironment')
    error('Reference file does not contain multiscale_microenvironment.');
end
if ~isfield(new_loaded, 'multiscale_microenvironment')
    error('New file does not contain multiscale_microenvironment.');
end

ref = ref_loaded.multiscale_microenvironment;
new = new_loaded.multiscale_microenvironment;

if ndims(ref) ~= 2 || ndims(new) ~= 2
    error('Expected both multiscale_microenvironment variables to be 2-D matrices.');
end
if size(ref, 1) < 5 || size(new, 1) < 5
    error('Expected at least 5 rows: x, y, z, volume, and one density row.');
end
if size(ref, 1) ~= size(new, 1) || size(ref, 2) ~= size(new, 2)
    error('Matrix sizes differ. Reference is %dx%d, new is %dx%d.', ...
        size(ref, 1), size(ref, 2), size(new, 1), size(new, 2));
end

[~, ref_order] = sortrows(ref(1:4, :)', [1, 2, 3, 4]);
[~, new_order] = sortrows(new(1:4, :)', [1, 2, 3, 4]);
ref = ref(:, ref_order);
new = new(:, new_order);

geometry_error = abs(ref(1:4, :) - new(1:4, :));
max_geometry_error = max(geometry_error(:));
if max_geometry_error > equivalence_tolerance
    error(['Files are not equivalent microenvironment layouts. ', ...
        'Maximum coordinate/volume error is %.17g, tolerance is %.17g.'], ...
        max_geometry_error, equivalence_tolerance);
end

num_voxels = size(ref, 2);
num_densities = size(ref, 1) - 4;

fprintf('\nMicroenvironment layouts are equivalent.\n');
fprintf('Voxels:    %d\n', num_voxels);
fprintf('Densities: %d\n', num_densities);
fprintf('Geometry max abs error: %.17g\n\n', max_geometry_error);

fprintf('%10s %18s %18s %18s %18s\n', ...
    'density', 'max_abs_error', 'RMSE', 'ref_max', 'new_max');
fprintf('%10s %18s %18s %18s %18s\n', ...
    '-------', '-------------', '----', '-------', '-------');

all_density_errors = ref(5:end, :) - new(5:end, :);
global_max_error = max(abs(all_density_errors(:)));
global_rmse = sqrt(mean(all_density_errors(:) .^ 2));

for row = 5:size(ref, 1)
    density_index = row - 4;
    density_error = ref(row, :) - new(row, :);
    max_abs_error = max(abs(density_error));
    rmse = sqrt(mean(density_error .^ 2));
    fprintf('%10d %18.10g %18.10g %18.10g %18.10g\n', ...
        density_index, max_abs_error, rmse, max(ref(row, :)), max(new(row, :)));
end

fprintf('\nGlobal density max abs error: %.17g\n', global_max_error);
fprintf('Global density RMSE:          %.17g\n', global_rmse);
