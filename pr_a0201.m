%% Part A 1
% List of file paths
mat_files = {
    "PR_CW_mat\cylinder_papillarray_single.mat", 
    "PR_CW_mat\cylinder_rubber_papillarray_single.mat", 
    "PR_CW_mat\cylinder_TPU_papillarray_single.mat", 
    "PR_CW_mat\hexagon_papillarray_single.mat", 
    "PR_CW_mat\hexagon_rubber_papillarray_single.mat", 
    "PR_CW_mat\hexagon_TPU_papillarray_single.mat"
};

% Create a figure for 3D trajectory plots
figure;

% Iterate through each file
for i = 1:length(mat_files)
    % Load data from the file
    data = load(mat_files{i});
    positions = data.end_effector_poses(:, 1:3); % Extract positions (x, y, z)
    n_points = size(positions, 1);               % Get the number of data points

    % Create a subplot for each file
    subplot(2, 3, i); % 2 rows, 3 columns layout, i-th subplot
    scatter3(positions(:, 1), positions(:, 2), positions(:, 3), 15, linspace(1, 10, n_points), 'filled');
    if i==3 || i==6
        colormap(jet);
        colorbar;
    end
    grid on;
    xlabel('X (m)');
    ylabel('Y (m)');
    zlabel('Z (m)');
    title(sprintf('Trajectory %d', i)); % Set the title
    axis equal;

end
% Create another figure for 2D top-view trajectories
figure;

% Iterate through each file
for i = 1:length(mat_files)
    % Load data from the file
    data = load(mat_files{i});
    poses = data.end_effector_poses;
    positions = poses(:, 1:3); % Extract positions (x, y, z)

    % Create a subplot for each file
    subplot(2, 3, i); % 2 rows, 3 columns layout, i-th subplot
    hold on;

    % Plot the trajectory in the XY plane
    plot(positions(:, 1), positions(:, 2), 'b', 'LineWidth', 1.5); % Plot X and Y only
    scatter(positions(:, 1), positions(:, 2), 15, linspace(1, 10, size(positions, 1)), 'filled'); % Add color-gradient points

    % Plot settings
    colormap(jet);
    grid on;
    xlabel('X (m)');
    ylabel('Y (m)');
    title(sprintf('Trajectory %d (Top View)', i)); % Set the title
    axis equal;

    % Set the view angle to top-down (XY plane)
    view(2); % 2 represents the XY plane top-down view
end

% Set a global title for the figure
%  sgtitle('Top-View Trajectories from Multiple Objects');

%% Part A 2
% List of file paths
mat_files = {
    "PR_CW_mat\cylinder_papillarray_single.mat", 
    "PR_CW_mat\cylinder_rubber_papillarray_single.mat", 
    "PR_CW_mat\cylinder_TPU_papillarray_single.mat", 
    "PR_CW_mat\hexagon_papillarray_single.mat", 
    "PR_CW_mat\hexagon_rubber_papillarray_single.mat", 
    "PR_CW_mat\hexagon_TPU_papillarray_single.mat",
    "PR_CW_mat\oblong_papillarray_single.mat",
    "PR_CW_mat\oblong_rubber_papillarray_single.mat",
    "PR_CW_mat\oblong_TPU_papillarray_single.mat"
};

% Create a figure for the subplots
figure;

% Iterate through each file
for i = 1:length(mat_files)
    % Load the data
    data = load(mat_files{i});
    ft_values = data.ft_values; % Force/torque sensor data
    sensor_matrices_displacement = data.sensor_matrices_displacement; % Tactile displacement data
    sensor_matrices_force = data.sensor_matrices_force; % Tactile force data

    % Extract force data
    forces = ft_values(:, 1:3); % Extract force values [x, y, z]
    z_force = forces(:, 3);     % Extract Z-direction force (normal contact force)
    
    % Find all peaks in the Z-direction force
    [~, locs] = findpeaks(abs(z_force), 'MinPeakProminence', 3); % Adjust MinPeakProminence as needed
    peaks = z_force(locs); % Extract peak values at the locations

    % Retrieve corresponding tactile and force/torque data for the peak indices
    tactile_forces = sensor_matrices_force(locs, :); % Tactile force data at peaks
    tactile_displacements = sensor_matrices_displacement(locs, :); % Tactile displacement data at peaks
    force_torques = ft_values(locs, :); % Force/torque data at peaks
    
    % Create a subplot for the current file
    subplot(3, 3, i); % 3x3 layout for 9 subplots
    plot(z_force, 'b-', 'LineWidth', 1.5); % Plot Z-direction force
    hold on;
    plot(locs, peaks, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r'); % Mark all peaks
    grid on;
    xlabel('Time Index');
    ylabel('Z Force (N)');
    
    % Set the subplot title to the filename
    [~, name, ~] = fileparts(mat_files{i}); % Extract the filename without path and extension
    title(name); % Use the filename as the subplot title
    
    % Save peaks and locs to a separate .mat file
    save_filename = fullfile('Processed_data\Peaks_and_locs', sprintf('%s.mat', name)); % Generate dynamic filename
    save(save_filename, 'peaks', 'locs'); % Save peaks and locs to .mat file
    
    % Save tactile data and force/torque data to a separate .mat file
    save_filename =  fullfile('Processed_data\Data_for_each_contact', sprintf('%s.mat', name));  % Generate dynamic filename
    save(save_filename, 'tactile_forces', 'tactile_displacements', 'force_torques', 'locs'); % Save data to .mat file
    
end

% Set the overall title for all subplots
sgtitle('Normal Contact Force and Detected Peaks (9 Objects)');



%% Part A 3
% List of file paths
mat_files = {
    "Processed_data\Data_for_each_contact\cylinder_papillarray_single.mat", 
    "Processed_data\Data_for_each_contact\cylinder_rubber_papillarray_single.mat", 
    "Processed_data\Data_for_each_contact\cylinder_TPU_papillarray_single.mat", 
};
% Define colors for each cylinder
colors = {'r', 'g', 'b'}; % Red for normal, Green for rubber, Blue for TPU

% Create a 3D scatter plot
figure;
hold on;
grid on;

% Iterate through each file
for i = 1:length(mat_files)
    data = load(mat_files{i});
    % Choose the papillae
    forces = data.tactile_forces(:, 13:15);  % middle
%     forces = data.tactile_forces(:, 1:3);  % corner
    % Scatter plot for this cylinder
    scatter3(forces(:, 1), forces(:, 2), forces(:, 3), 50, colors{i}, 'filled'); 
end

% Add labels, legend, and title
xlabel('Force in X direction (N)');
ylabel('Force in Y direction (N)');
zlabel('Force in Z direction (N)');
title('3D Scatter Plot of Tactile Forces');
legend({'Normal', 'Rubber', 'TPU'}, 'Location', 'Best');
view(3);

