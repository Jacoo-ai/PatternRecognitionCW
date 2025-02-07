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

% Calcualte eigenvalues and sort
[V, D] = eig(Sw \ Sb);
[eigenvalues, index] = sort(diag(D), 'descend');
W_LDA = V(:, index);

% Project data to 2D
LD1 = W_LDA(:, 1);
LD2 = W_LDA(:, 2);
X_lda_2D = X * [LD1, LD2];

% Plot 2D data points
figure; hold on;
X_LD1 = X_lda_2D(:, 1);
X_LD2 = X_lda_2D(:, 2);
s1 = scatter(X_LD1(y == 1), X_LD2(y == 1), 'g', 'filled'); % Rubber
s2 = scatter(X_LD1(y == 2), X_LD2(y == 2), 'b', 'filled'); % TPU

% Plot LDA directions (LD1 & LD2) 
q1 = quiver(mean(X_LD1), mean(X_LD2), LD1(1), LD1(2), 1.5, 'r', 'LineWidth', 2, 'MaxHeadSize', 0.5); % LD1
q2 = quiver(mean(X_LD1), mean(X_LD2), LD2(1), LD2(2), 1.5, 'm', 'LineWidth', 2, 'MaxHeadSize', 0.5); % LD2

% Plot 2D LDA decision boundary
decision_boundary = (mu1 + mu2) / 2; % Midpoint between class means
boundary_slope = -LD1(1) / LD1(2); % Perpendicular slope
x_vals = linspace(min(X_lda_2D(:, 1)), max(X_lda_2D(:, 1)), 10);
y_vals = boundary_slope * (x_vals - decision_boundary(1)) + decision_boundary(2);
decision_boundary_line = plot(x_vals, y_vals, '--k', 'LineWidth', 2); % Black dashed line

% Labels and title
xlabel('LD1');
ylabel('LD2');
xlim([min(X_LD1)-1, max(X_LD1)+1]); % Limit x axis range
ylim([min(X_LD2)-1, max(X_LD2)+1]); % Limit y axis range
legend([s1, s2, q1, q2, decision_boundary_line], ...
       {'Rubber', 'TPU', 'LD1', 'LD2', 'Decision Boundary'}, ...
       'Location', 'best');
title('2D LDA Projection with Classification Boundary');
grid on;
axis equal;
hold off;


% Create figure for the 3D LDA plane
figure; hold on;

% Scatter plot of original 3D data points
s1 = scatter3(X(y == 1, 1), X(y == 1, 2), X(y == 1, 3), 20, 'g', 'filled'); % Rubber
s2 = scatter3(X(y == 2, 1), X(y == 2, 2), X(y == 2, 3), 20, 'b', 'filled'); % TPU

% Define the range for the plane in LD1-LD2 space
plane_size = 5;
ld1_range = linspace(-plane_size, plane_size, 10);
ld2_range = linspace(-plane_size, plane_size, 10);
[LD1_grid, LD2_grid] = meshgrid(ld1_range, ld2_range);

% Compute corresponding 3D points for the plane
X_plane = mu_total(1) + LD1(1) * LD1_grid + LD2(1) * LD2_grid;
Y_plane = mu_total(2) + LD1(2) * LD1_grid + LD2(2) * LD2_grid;
Z_plane = mu_total(3) + LD1(3) * LD1_grid + LD2(3) * LD2_grid;

% Plot LDA plane using surf()
lda_plane = surf(X_plane, Y_plane, Z_plane, 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'FaceColor', 'magenta');

% Labels and legend
xlabel('D_X');
ylabel('D_Y');
zlabel('D_Z');
legend([s1, s2, lda_plane], {'Rubber', 'TPU', 'LDA Projection Plane'}, 'Location', 'best');

% Set title and grid
title('3D LDA Projection with Classification Plane');
grid on;
view(3); % 3D view
axis equal;
hold off;
