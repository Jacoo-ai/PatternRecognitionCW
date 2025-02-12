%% Part C a,b - 3D Plot of Tactile Displacement for Different Materials
clear;
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
    [~, index] = sort(diag(D), 'descend');
    W_LDA = V(:, index);
    
    % Extract first LDA direction (LD1)
    LD1 = W_LDA(:, 1);
    
    % Scatter plot of original data
    scatter(X(y == 1, 1), X(y == 1, 2), 20, 'g', 'filled'); % Rubber
    scatter(X(y == 2, 1), X(y == 2, 2), 20, 'b', 'filled'); % TPU
    
    % Normalize the LDA direction vector
    LD1 = LD1 / norm(LD1);
    
    % Compute the mean of the data
    mu_total = mean(X, 1);
    
    % Set the line range (adjust factor to fit data)
    s = 3;  % Control the length of the line
    P1 = mu_total + s * LD1';  % Endpoint in one direction
    P2 = mu_total - s * LD1';  % Endpoint in the opposite direction

    % Plot the LDA direction line
    plot([P1(1), P2(1)], [P1(2), P2(2)], 'r', 'LineWidth', 2);
    
    % Compute the mean projections of each class onto the LDA direction
    mu1_proj = mean(X(y == 1, :) * LD1);  % Mean projection for class 1 (Rubber)
    mu2_proj = mean(X(y == 2, :) * LD1);  % Mean projection for class 2 (TPU)
    
    % Compute the decision boundary (midpoint in the LDA direction)
    decision_boundary_proj = (mu1_proj + mu2_proj) / 2;
    
    % Compute the decision boundary point
    decision_boundary_point = mu_total + decision_boundary_proj * LD1';
    
    % Compute the direction perpendicular to LD1
    LD1_perp = [-LD1(2), LD1(1)];  % Rotate by 90 degrees
    
    % Compute endpoints of the decision boundary line
    B1 = decision_boundary_point + s * LD1_perp;
    B2 = decision_boundary_point - s * LD1_perp;

    % Plot the decision boundary as a dashed black line
    plot([B1(1), B2(1)], [B1(2), B2(2)], 'k--', 'LineWidth', 2);
    
    % Labels and title
    xlabel(feature_labels{features(1)});
    ylabel(feature_labels{features(2)});
    title(sprintf('LDA on %s vs %s', feature_labels{features(1)}, feature_labels{features(2)}));
    legend({'Rubber', 'TPU', 'LDA Direction', 'Decision Boundary'}, 'Location', 'best');
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

% Calcualte eigenvalues and sort
[V, D] = eig(Sw \ Sb);
[eigenvalues, index] = sort(diag(D), 'descend');
W_LDA = V(:, index);

% Project data to 2D
LD1 = W_LDA(:, 1);
LD2 = W_LDA(:, 2);
X_lda_2D = X * [LD1, LD2];

% Ensure projected data don't flip because of the direction of the eigen vector
if mean(X_lda_2D(y == 1, 1)) > mean(X_lda_2D(y == 2, 1))
    W_LDA(:,1) = -W_LDA(:,1);
    X_lda_2D(:,1) = -X_lda_2D(:,1);
end
if mean(X_lda_2D(y == 1, 2)) < mean(X_lda_2D(y == 2, 2))
    W_LDA(:,2) = -W_LDA(:,2);
    X_lda_2D(:,2) = -X_lda_2D(:,2);
end

% Plot 2D data points
figure; hold on;
X_LD1 = X_lda_2D(:, 1);
X_LD2 = X_lda_2D(:, 2);
scatter(X_LD1(y == 1), X_LD2(y == 1), 'g', 'filled'); % Rubber
scatter(X_LD1(y == 2), X_LD2(y == 2), 'b', 'filled'); % TPU

% Extract LD1 and LD2 components
LD1_unit = LD1(1:2)'; % Convert to row vector
LD1_unit = LD1_unit / norm(LD1_unit); % Normalize

LD2_unit = LD2(1:2)'; % Convert to row vector
LD2_unit = LD2_unit / norm(LD2_unit); % Normalize

% Compute the mean of the projected data
mu_lda = mean(X_lda_2D, 1);

% Set scale for visualization
s = 3;

% Compute LD1 and LD2 lines
P1_LD1 = mu_lda + s * LD1_unit; % Endpoint in one direction
P2_LD1 = mu_lda - s * LD1_unit; % Endpoint in the opposite direction

P1_LD2 = mu_lda + s * LD2_unit; % Endpoint in one direction
P2_LD2 = mu_lda - s * LD2_unit; % Endpoint in the opposite direction

% Plot LD1 and LD2 as solid lines
plot([P1_LD1(1), P2_LD1(1)], [P1_LD1(2), P2_LD1(2)], 'r', 'LineWidth', 2); % LD1 direction
plot([P1_LD2(1), P2_LD2(1)], [P1_LD2(2), P2_LD2(2)], 'm', 'LineWidth', 2); % LD2 direction

% Projected class means onto LD1
mu1_proj = mean(X_lda_2D(y == 1, :) * LD1_unit');
mu2_proj = mean(X_lda_2D(y == 2, :) * LD1_unit');

% Compute decision boundary position (midpoint of projected means)
decision_boundary_proj = (mu1_proj + mu2_proj) / 2;
decision_boundary_point = mu_lda + decision_boundary_proj * LD1_unit;

% Compute perpendicular direction to LD1
LD1_perp = [-LD1_unit(2), LD1_unit(1)];  % Rotate 90 degrees

% Compute endpoints of the decision boundary line
B1 = decision_boundary_point + s * LD1_perp;
B2 = decision_boundary_point - s * LD1_perp;

% Plot decision boundary as a dashed black line
plot([B1(1), B2(1)], [B1(2), B2(2)], 'k--', 'LineWidth', 2);

% Labels and title
xlabel('LD1');
ylabel('LD2');
xlim([min(X_LD1)-1, max(X_LD1)+1]); % Adjust x-axis limits
ylim([min(X_LD2)-1, max(X_LD2)+1]); % Adjust y-axis limits
legend({'Rubber', 'TPU', 'LD1', 'LD2', 'Decision Boundary'}, 'Location', 'best');
title('2D LDA Projection with Decision Boundary');
grid on;
axis equal;
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
grid on;
view(3); % 3D view
axis equal;
hold off;
