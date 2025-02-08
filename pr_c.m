%% Part C a,b - 3D Plot of Tactile Displacement for Different Materials

% List of file paths
mat_files = {
    "Processed_data\Data_for_each_contact\oblong_rubber_papillarray_single.mat", 
    "Processed_data\Data_for_each_contact\oblong_TPU_papillarray_single.mat"
};

% Define colors for each material
colors = {'g', 'b'}; % Green for rubber, Blue for TPU

% Initialize a matrix to hold all displacements
all_displacements = [];
displacement_labels = []; % Store material labels for coloring

% Iterate through each file to load displacement data
for i = 1:length(mat_files)
    data = load(mat_files{i});
    % Choose the middle sensor's displacement data (columns 13:15)
    displacements = data.tactile_displacements(:, 13:15);
    % Append displacements to the matrix
    all_displacements = [all_displacements; displacements];
    displacement_labels = [displacement_labels; i * ones(size(displacements, 1), 1)];
end

% Create a figure for the 3D scatter plot
figure; hold on;

% Plot each material separately with different colors
for i = 1:length(mat_files)
    scatter3(all_displacements(displacement_labels == i, 1), ...
             all_displacements(displacement_labels == i, 2), ...
             all_displacements(displacement_labels == i, 3), ...
             20, colors{i}, 'filled');
end

% Set plot labels, legend, and title
xlabel('X Displacement');
ylabel('Y Displacement');
zlabel('Z Displacement');
title('3D Plot of the Tactile Displacement for Different Materials');
grid on;
view(3);

% Add legend
legend({'Rubber Cylinder', 'TPU Cylinder'}, 'Location', 'Best');

hold off;



%% Part C c - Apply LDA to all 2D combinations of D_X, D_Y, and D_Z

% Define colors for each material
colors = {'g', 'b'}; % Green for rubber, Blue for TPU

% Define 2D feature combinations
feature_combinations = {[1, 2], [1, 3], [2, 3]};
feature_labels = {'D_X', 'D_Y', 'D_Z'};

% Create figure for subplots
figure;

for m = 1:length(feature_combinations)
    subplot(1, 3, m); hold on;
    
    % Select feature indices
    features = feature_combinations{m};
    X = all_displacements(:, features);  % Select two features (e.g., D_X and D_Y)
    y = displacement_labels;  % Class labels (1 for Rubber, 2 for TPU)
    
    % Standardization (optional, can be removed if not needed)
    X = (X - mean(X)) ./ std(X); 
    
    % Calculate means
    mu1 = mean(X(y == 1, :), 1);
    mu2 = mean(X(y == 2, :), 1);
    mu_total = mean(X, 1);
    
    % Calculate Sw and Sb
    Sw = cov(X(y == 1, :)) + cov(X(y == 2, :));
    Sb = (mu1 - mu_total)' * (mu1 - mu_total) + (mu2 - mu_total)' * (mu2 - mu_total);
    
    % Compute LDA eigenvectors
    [V, D] = eig(Sw \ Sb);
    [eigenvalues, index] = sort(diag(D), 'descend');
    W_LDA = V(:, index);
    
    % Extract first LDA direction (LD1)
    LD1 = W_LDA(:, 1);
    
    % Scatter plot of original data
    scatter(X(y == 1, 1), X(y == 1, 2), 20, 'g', 'filled'); % Rubber
    scatter(X(y == 2, 1), X(y == 2, 2), 20, 'b', 'filled'); % TPU

    % Draw LDA direction as an arrow from the total mean
    quiver(mu_total(1), mu_total(2), LD1(1), LD1(2), 2, 'r', 'LineWidth', 2, 'MaxHeadSize', 0.5);
     
    % Compute perpendicular decision boundary line
    decision_boundary = (mu1 + mu2) / 2; % Midpoint between class means
    boundary_slope = -LD1(1) / LD1(2); % Perpendicular slope
    x_vals = linspace(min(X(:, 1)), max(X(:, 1)), 100);
    y_vals = boundary_slope * (x_vals - decision_boundary(1)) + decision_boundary(2);
    plot(x_vals, y_vals, '--k', 'LineWidth', 2); % Black dashed line

    % Labels and title
    xlabel(feature_labels{features(1)});
    ylabel(feature_labels{features(2)});
    xlim([min(X(:, 1))-1, max(X(:, 1))+1]); % Limit x axis range
    ylim([min(X(:, 2))-1, max(X(:, 2))+1]); % Limit y axis range
    title(['LDA on ', feature_labels{features(1)}, ' vs ', feature_labels{features(2)}]);
    legend({'Rubber', 'TPU', 'LD1', 'Decision Boundary'}, 'Location', 'best');
    grid on;
end
hold off;




%% Part C d - Apply LDA to the 3D displacement data

% Define colors for each material
colors = {'g', 'b'}; % Green for rubber, Blue for TPU

% Prepare 3D displacement data
X = all_displacements;  % Feature matrix (D_X, D_Y, D_Z)
y = displacement_labels;  % Class labels (1 for Rubber, 2 for TPU)

% Standardization
X = (X - mean(X)) ./ std(X); 

% Calculate means
mu1 = mean(X(y == 1, :), 1);
mu2 = mean(X(y == 2, :), 1);
mu_total = mean(X, 1);

% Calculate Sw and Sb
Sw = cov(X(y == 1, :)) + cov(X(y == 2, :));
Sb = (mu1 - mu_total)' * (mu1 - mu_total) + (mu2 - mu_total)' * (mu2 - mu_total);

% Compute eigenvalues and sort
[V, D] = eig(Sw \ Sb);
[eigenvalues, index] = sort(diag(D), 'descend');
W_LDA = V(:, index);

% Get LD1 and LD2 (LDA projection plane)
LD1 = W_LDA(:, 1);
LD2 = W_LDA(:, 2);

% Compute decision boundary (midpoint of class means)
decision_boundary = (mu1 + mu2) / 2;

% Create figure for the 3D classification plane
figure; hold on;

% Scatter plot of original 3D data points
s1 = scatter3(X(y == 1, 1), X(y == 1, 2), X(y == 1, 3), 20, 'g', 'filled'); % Rubber
s2 = scatter3(X(y == 2, 1), X(y == 2, 2), X(y == 2, 3), 20, 'b', 'filled'); % TPU

% Define the range for the classification plane
plane_size = 5;
x_range = linspace(min(X(:,1)), max(X(:,1)), 10);
y_range = linspace(min(X(:,2)), max(X(:,2)), 10);
[X_grid, Y_grid] = meshgrid(x_range, y_range);

% Compute classification plane normal (LD1)
normal = LD1;  % classification is vertical to LD1

% Compute corresponding Z values for classification plane
Z_grid = (-normal(1) * (X_grid - decision_boundary(1)) ...
          - normal(2) * (Y_grid - decision_boundary(2))) / normal(3) ...
          + decision_boundary(3);

% Plot classification plane using surf()
classification_plane = surf(X_grid, Y_grid, Z_grid, ...
                            'FaceAlpha', 0.5, 'EdgeColor', 'none', 'FaceColor', 'cyan');

% Labels and legend
xlabel('D_X');
ylabel('D_Y');
zlabel('D_Z');
legend([s1, s2, classification_plane], {'Rubber', 'TPU', 'Classification Plane'}, 'Location', 'best');

% Set title and grid
title('3D Classification Plane (Decision Boundary)');
grid on;
view(3); % 3D view
axis equal;
hold off;
