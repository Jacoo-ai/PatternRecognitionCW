%% Part B 1a
clc; clear; close all;
% List of file paths
mat_files = {
    "Processed_data\Data_for_each_contact\cylinder_papillarray_single.mat", 
    "Processed_data\Data_for_each_contact\cylinder_rubber_papillarray_single.mat", 
    "Processed_data\Data_for_each_contact\cylinder_TPU_papillarray_single.mat"
};

% Define colors for each cylinder
colors = {'r', 'g', 'b'}; % Red for normal, Green for rubber, Blue for TPU

% Initialize a matrix to hold all forces from the three objects
all_forces = [];
force_labels = []; % Store material labels for coloring
data_per_material = {}; % Store standardized forces separately

% Iterate through each file to load force data
for i = 1:length(mat_files)
    data = load(mat_files{i});
    % Choose the middle sensor's force data (columns 13:15)
    forces = data.tactile_forces(:, 13:15);
    % Append forces to all_forces matrix
    all_forces = [all_forces; forces];
    force_labels = [force_labels; i * ones(size(forces, 1), 1)];
end

% Step 1: Standardize the force data
standardized_forces = (all_forces - mean(all_forces)) ./ std(all_forces);

% Step 2: Apply PCA
[coeff, score, ~] = pca(standardized_forces); % coeff: principal components, score: transformed data

% Step 3: Replot the standardized data with principal components
figure; hold on;

% Plot each material separately with different colors
for i = 1:length(mat_files)
    scatter3(standardized_forces(force_labels == i, 1), ...
             standardized_forces(force_labels == i, 2), ...
             standardized_forces(force_labels == i, 3), ...
             20, colors{i}, 'filled');
end

% Plot the principal components as vectors
mean_force = mean(standardized_forces);
quiver3(mean_force(1), mean_force(2), mean_force(3), ...
        coeff(1, 1), coeff(2, 1), coeff(3, 1), 5, 'Color', [1 0.5 0], 'LineWidth', 2, 'MaxHeadSize', 0.5); % PC1 (orange)
quiver3(mean_force(1), mean_force(2), mean_force(3), ...
        coeff(1, 2), coeff(2, 2), coeff(3, 2), 5, 'Color', [0.5 0 0.5], 'LineWidth', 2, 'MaxHeadSize', 0.5); % PC2 (purple)
quiver3(mean_force(1), mean_force(2), mean_force(3), ...
        coeff(1, 3), coeff(2, 3), coeff(3, 3), 5, 'Color', [0 0.75 0.75], 'LineWidth', 2, 'MaxHeadSize', 0.5); % PC3 (cyan)

% Set plot labels, legend, and title
xlabel('Standardised Force X');
ylabel('Standardised Force Y');
zlabel('Standardised Force Z');
title('PCA of Standardised Force Data with Principal Components');

legend({'Normal Cylinder', 'Rubber Cylinder', 'TPU Cylinder', 'PC1', 'PC2', 'PC3'}, 'Location', 'Best');
grid on; box on;
view(3);
hold off;




%% Part B 1b

% Reduce to 2D using first two principal components
reduced_data = score(:, 1:2);

% Replot the standardized data in 2D
figure; hold on;

% Plot each material separately with different colors
for i = 1:length(mat_files)
    scatter(reduced_data(force_labels == i, 1), ...
            reduced_data(force_labels == i, 2), ...
            20, colors{i}, 'filled');
end

% Set plot labels, legend, and title
xlabel('PC1');
ylabel('PC2');
title('PCA of Standardized Force Data (2D)');

legend({'Normal Cylinder', 'Rubber Cylinder', 'TPU Cylinder'}, 'Location', 'Best');
grid on; box on;
hold off;



%% Part B 1c - Plot distributions on 1D number lines

% Create a figure for 1D number lines
figure;

% Define colors for each material
colors = {'r', 'g', 'b'}; % Red for normal, Green for rubber, Blue for TPU

for pc = 1:3
    subplot(1, 3, pc); % 3 PCs per row
    hold on;

    % Plot each material separately with different colors
    for i = 1:length(mat_files)
        scatter(score(force_labels == i, pc), ...
                zeros(sum(force_labels == i), 1), ...
                15, colors{i}, 'filled');
    end
    
    % Set labels, title, and grid
    xlabel(sprintf('PC%d', pc));
    ylabel('');
    title(sprintf('PC%d', pc));
    grid on; box on;
    legend({'Normal Cylinder', 'Rubber Cylinder', 'TPU Cylinder'}, 'Location', 'best');
    ylim([-0.5, 0.5]); % Make space for clarity
    hold off;
end

% Set an overall title for the figure
sgtitle('1D Distribution for the Middle Papillae');




%% Part B 2a

mat_files = {
    "Processed_data\Data_for_each_contact\cylinder_papillarray_single.mat", 
    "Processed_data\Data_for_each_contact\cylinder_rubber_papillarray_single.mat", 
    "Processed_data\Data_for_each_contact\cylinder_TPU_papillarray_single.mat"
};

% Initialize a matrix to hold all forces from the three objects
all_forces = [];

% Iterate through each file to load force data
for i = 1:length(mat_files)
    data = load(mat_files{i});
    forces = data.tactile_forces; % Use all tactile sensor's force data
    all_forces = [all_forces; forces]; % Append forces to all_forces matrix
end

% Step 1: Standardize the force data
standardized_forces = (all_forces - mean(all_forces)) ./ std(all_forces);

% Step 2: Apply PCA
[~, score, latent, ~, explained] = pca(standardized_forces);

% Step 3: Find the elbow point using the 'knee' method
diff_exp = diff(explained); % Compute first derivative
second_diff_exp = diff(diff_exp); % Compute second derivative
[~, elbow] = max(abs(second_diff_exp)); % Elbow is at max curvature

% Step 4: Create a scree plot
figure;
hold on;
plot(1:length(explained), explained, 'bo-', 'LineWidth', 1.5); % Line and points
plot([0, length(explained)], [explained(elbow+1), explained(elbow+1)], 'r--', 'LineWidth', 1.5); % Elbow vertical line

% Formatting
grid on; box on;
xlabel('Principal Component');
ylabel('Variance Explained (%)');
xlim([0 15]);
title('Scree Plot');
hold off;



%% Part B 2b - 1D Distribution for 9 Sensors with Material Separation

% Define colors for each material
colors = {'r', 'g', 'b'}; % Red for normal, Green for rubber, Blue for TPU

% Create plot
figure;

% Plot the first 3 PCs
for pc = 1:3
    subplot(1, 3, pc); % Create a subplot
    hold on;
    
    % Plot each material separately with different colors
    for i = 1:length(mat_files)
        scatter(score(force_labels == i, pc), ...
                zeros(sum(force_labels == i), 1), ...
                15, colors{i}, 'filled'); % 1D scatter plot for each material
    end
    
    xlabel(sprintf('PC%d', pc));
    ylabel('');
    title(sprintf('PC%d', pc));
    legend({'Normal Cylinder', 'Rubber Cylinder', 'TPU Cylinder'}, 'Location', 'north');
    grid on; box on;
    ylim([-0.5, 0.5]); % Make space for clarity
    hold off;
end

% Set an overall title for the figure
sgtitle('1D Distribution for 9 Papillae');




%% Part B 2c - 2D PCA Reduction for 9 Sensors with Material Separation

% Create figure
figure; hold on;

% Reduce to 2D (PC1 and PC2)
reduced_data = score(:, 1:2);

% Define colors for each material
colors = {'r', 'g', 'b'}; % Red for normal, Green for rubber, Blue for TPU

% Plot each material separately with different colors
for i = 1:length(mat_files)
    scatter(reduced_data(force_labels == i, 1), ...
            reduced_data(force_labels == i, 2), ...
            20, colors{i}, 'filled');
end

% Set labels and title
xlabel('PC1');
ylabel('PC2');
title('2D PCA Reduction for 9 Sensors');

% Add legend
legend({'Normal Cylinder', 'Rubber Cylinder', 'TPU Cylinder'}, 'Location', 'Best');

grid on; box on;
hold off;