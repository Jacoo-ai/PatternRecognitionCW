%% Part B 1a

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

% Iterate through each file to load force data
for i = 1:length(mat_files)
    data = load(mat_files{i});
    % Choose the middle sensor's force data (columns 13:15)
    forces = data.tactile_forces(:, 13:15);
    % Append forces to all_forces matrix
    all_forces = [all_forces; forces];
end

% Step 1: Standardize the force data
standardized_forces = (all_forces - mean(all_forces)) ./ std(all_forces);

% Step 2: Apply PCA
[coeff, score, ~] = pca(standardized_forces); % coeff: principal components, score: transformed data

% Step 3: Replot the standardized data with principal components
figure;
scatter3(standardized_forces(:, 1), standardized_forces(:, 2), standardized_forces(:, 3), 20, 'k', 'filled');
hold on;

% Plot the principal components as vectors
mean_force = mean(standardized_forces);
quiver3(mean_force(1), mean_force(2), mean_force(3), ...
        coeff(1, 1), coeff(2, 1), coeff(3, 1), 5, 'r', 'LineWidth', 2, 'MaxHeadSize', 0.5); % PC1
quiver3(mean_force(1), mean_force(2), mean_force(3), ...
        coeff(1, 2), coeff(2, 2), coeff(3, 2), 5, 'g', 'LineWidth', 2, 'MaxHeadSize', 0.5); % PC2
quiver3(mean_force(1), mean_force(2), mean_force(3), ...
        coeff(1, 3), coeff(2, 3), coeff(3, 3), 5, 'b', 'LineWidth', 2, 'MaxHeadSize', 0.5); % PC3

% Set plot labels, legend, and title
xlabel('Standardized Force X');
ylabel('Standardized Force Y');
zlabel('Standardized Force Z');
title('PCA of Standardized Force Data with Principal Components');
legend({'Standardized Data', 'PC1', 'PC2', 'PC3'}, 'Location', 'Best');
grid on;
view(3);



%% Part B 1b

% Create a figure for 2D plots
figure;

% Reduce to 2D (PC1 and PC2)
reduced_data = score(:, 1:2);

% Plot 2D data
scatter(reduced_data(:, 1), reduced_data(:, 2), 20,"black", 'filled');
xlabel('PC1');
ylabel('PC2');

% Set an overall title for the figure
sgtitle('2D PCA Reduction for the Middle Sensor');



%% Part B 1c - Plot distributions on 1D number lines

% Create a figure for 1D number lines
figure;

for pc = 1:3
        subplot(1, 3, pc); % 3 PCs per row
        scatter(score(:, pc), zeros(size(score, 1), 1), 15, "black", 'filled'); % 1D scatter plot
        
        % Set labels, title, and grid
        xlabel(sprintf('PC%d', pc));
        ylabel('');
        title(sprintf('PC%d', pc));
        grid on;
        ylim([-0.5, 0.5]); % Make space for clarity
end

% Set an overall title for the figure
sgtitle('1D Distribution for the Middle Sensor');




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
    % Use all tactile sensor's force data
    forces = data.tactile_forces;
    % Append forces to all_forces matrix
    all_forces = [all_forces; forces];
end

% Step 1: Standardize the force data
standardized_forces = (all_forces - mean(all_forces)) ./ std(all_forces);

% Step 2: Apply PCA
[~, score, ~, ~, explained] = pca(standardized_forces);

% Step 3: Create a plot
figure;
bar(explained, 'FaceColor', [0.2, 0.6, 0.8]); % Bar figure
grid on;
xlabel('Principal Component');
ylabel('Variance Explained (%)');
title(sprintf('Scree Plot for 9 Sensors')); % Title




%% Part B 2b

% Create plot
figure;
% Plot 3 PCs
for pc = 1:3
    subplot(1, 3, pc); % Create a subplot
    scatter(score(:, pc), zeros(size(score, 1), 1), 15, "black", 'filled');
    xlabel(sprintf('PC%d', pc));
    ylabel('');
    title(sprintf('PC%d', pc));
    grid on;
    ylim([-0.5, 0.5]); % % Make space for clarity
end

% Set an overall title for the figure
sgtitle('1D Distribution for 9 Sensors');



%% Part B 2c
% Create plot
figure;

% Reduce to 2D (PC1 and PC2)
reduced_data = score(:, 1:2);

% Plot
scatter(reduced_data(:, 1), reduced_data(:, 2), 20, 'b', 'filled');
xlabel('PC1');
ylabel('PC2');
title(sprintf('2D PCA Reduction for 9 sensors')); % Title

