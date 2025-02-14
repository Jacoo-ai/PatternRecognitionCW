%% Part C a,b - 3D Plot of Tactile Displacement for Different Materials
clear; clc; close all;
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
grid on; box on;
view(3);

% Add legend
legend({'Rubber Cylinder', 'TPU Cylinder'}, 'Location', 'Best');

hold off;



%% Part C c - Apply LDA to all 2D combinations of D_X, D_Y, and D_Z
% Standardize the data
disp = zscore(all_displacements);
disp_sel = disp(:, 1:3); % Select D_X, D_Y, and D_Z

% Compute class means
classes = unique(displacement_labels);
mu = zeros(length(classes), size(disp, 2));
for i = 1:length(classes)
    mu(i, :) = mean(disp(displacement_labels == classes(i), :), 1);
end

% Compute Within-Class Scatter Matrix
Sw = zeros(size(disp, 2));
for i = 1:length(classes)
    class_data = disp(displacement_labels == classes(i), :);
    Si = cov(class_data);
    Sw = Sw + Si;
end

% Compute Between-Class Scatter Matrix
mu_overall = mean(disp, 1);
Sb = zeros(size(disp, 2));
for i = 1:length(classes)
    Ni = sum(displacement_labels == classes(i));
    diff = (mu(i, :) - mu_overall)';
    Sb = Sb + Ni * (diff * diff');
end

% Solve eigenvalue problem
[V, D] = eig(pinv(Sw) * Sb);

% Sort eigenvalues and select top eigenvectors
[eigenvalues_sorted, idx] = sort(diag(D), 'descend');
W = V(:, idx);

% Define feature labels
feature_labels = {'D_X', 'D_Y', 'D_Z'};
feature_combinations = [1 2; 1 3; 2 3];  % Combinations for (D_X, D_Y), (D_X, D_Z), (D_Y, D_Z)

figure;
for i = 1:3
    % Select the feature pair
    x_idx = feature_combinations(i, 1);
    y_idx = feature_combinations(i, 2);
    disp_2d = disp_sel(:, [x_idx, y_idx]);

    % Compute LDA projection direction
    lda_vector = W([x_idx, y_idx], 1); 
    proj_scale = (disp_2d - mean(disp_2d, 1)) * lda_vector / (lda_vector' * lda_vector);
    disp_projected = mean(disp_2d, 1) + proj_scale * lda_vector';

    % Plot scatter data
    subplot(1, 3, i); hold on;
    scatter(disp_sel(displacement_labels == 1, x_idx), disp_sel(displacement_labels == 1, y_idx), 20, 'g', 'filled', 'DisplayName', 'Rubber');
    scatter(disp_sel(displacement_labels == 2, x_idx), disp_sel(displacement_labels == 2, y_idx), 20, 'b', 'filled', 'DisplayName', 'TPU');
    scatter(disp_projected(displacement_labels == 1, 1), disp_projected(displacement_labels == 1, 2), 20, 'g', 'o', 'DisplayName', 'Projected Rubber');
    scatter(disp_projected(displacement_labels == 2, 1), disp_projected(displacement_labels == 2, 2), 20, 'b', 'o', 'DisplayName', 'Projected TPU');

    % Compute LDA direction line
    xlim([-4, 4]); 
    ylim([-4, 4]);
    x_range = xlim;
    y_range = ylim;
    x_vals = linspace(x_range(1), x_range(2), 100);
    y_vals = mean(disp_2d(:, 2)) + (lda_vector(2) / lda_vector(1)) * (x_vals - mean(disp_2d(:, 1)));
    plot(x_vals, y_vals, 'r-', 'LineWidth', 1.0, 'DisplayName', 'LDA Direction');

    % Compute decision boundary
    center_point = mean(disp_projected, 1);                 
    discrimination_slope = -lda_vector(1) / lda_vector(2);  
    x_decision_vals = linspace(x_range(1), x_range(2), 100);
    y_decision_vals = center_point(2) + discrimination_slope * (x_decision_vals - center_point(1));

    % Plot decision boundary
    plot(x_decision_vals, y_decision_vals, 'k--', 'LineWidth', 1.0, 'DisplayName', 'Decision Boundary');

    % Labels and title
    title(['LDA on ', feature_labels{x_idx}, ' vs ', feature_labels{y_idx}]);
    xlabel(feature_labels{x_idx});
    ylabel(feature_labels{y_idx});
    legend('Location', 'SouthWest');
    grid on; box on;
end
hold off;





%% Part C d - Apply LDA to the 3D displacement data

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

% Calculate eigenvalues and sort
[V, D] = eig(Sw \ Sb);
[eigenvalues, index] = sort(diag(D), 'descend');
W_LDA = V(:, index);

% Project data to 2D
LD1 = W_LDA(:, 1);
LD2 = W_LDA(:, 2);
X_lda_2D = X * [LD1, LD2];

% Ensure projected data consistency
if mean(X_lda_2D(y == 1, 1)) > mean(X_lda_2D(y == 2, 1))
    W_LDA(:,1) = -W_LDA(:,1);
    X_lda_2D(:,1) = -X_lda_2D(:,1);
end
if mean(X_lda_2D(y == 1, 2)) < mean(X_lda_2D(y == 2, 2))
    W_LDA(:,2) = -W_LDA(:,2);
    X_lda_2D(:,2) = -X_lda_2D(:,2);
end

% Compute exact rotation angle to align LD1 horizontally
theta = atan2(LD1(2), LD1(1));  % Compute rotation angle from LD1 direction
R = [cos(-theta), -sin(-theta); sin(-theta), cos(-theta)];  % Rotation matrix
X_lda_2D = X_lda_2D * R';

% Plot 2D data points
figure; hold on;
xlim([-3 3]);
ylim([-3 3]);
X_LD1 = X_lda_2D(:, 1);
X_LD2 = X_lda_2D(:, 2);
scatter(X_LD1(y == 1), X_LD2(y == 1), 'g', 'filled'); % Rubber
scatter(X_LD1(y == 2), X_LD2(y == 2), 'b', 'filled'); % TPU

% Compute class means in rotated space
mu1_proj = mean(X_lda_2D(y == 1, :), 1);
mu2_proj = mean(X_lda_2D(y == 2, :), 1);

% Compute decision boundary (midpoint of means)
decision_boundary_proj = (mu1_proj + mu2_proj) / 2;

% Decision boundary should be vertical (constant X value)
x_decision = decision_boundary_proj(1);

% **Extend the decision boundary beyond the current ylim**
ylimits = ylim;  % Get current y-axis limits
plot([x_decision, x_decision], [ylimits(1) - 1, ylimits(2) + 1], 'k--', 'LineWidth', 1);

% Labels and title
xlabel('LD1');
ylabel('LD2');
legend({'Rubber', 'TPU', 'Decision Boundary'}, 'Location', 'best');
title('2D LDA Projection with Extended Decision Boundary');
grid on; box on;
hold off;




% Create figure for the 3D LDA plane
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
normal = LD1;  % vertical to LD1

% Compute corresponding Z values for classification plane
Z_grid = (-normal(1) * (X_grid - decision_boundary_point(1)) ...
          - normal(2) * (Y_grid - decision_boundary_point(2))) / normal(3) ...
          +  mean(X(:, 3));

% Plot classification plane using surf()
classification_plane = surf(X_grid, Y_grid, Z_grid, ...
                            'FaceAlpha', 0.5, 'EdgeColor', 'none', 'FaceColor', 'cyan');
% Labels and legend
xlabel('D_X');
ylabel('D_Y');
zlabel('D_Z');
legend([s1, s2, classification_plane], {'Rubber', 'TPU', 'Classification Plane'}, 'Location', 'best');

% Set title and grid
title('3D LDA Projection with Classification Plane');
grid on; box on;
view(3); % 3D view
axis equal;
hold off;