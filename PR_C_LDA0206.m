clc; clear; close all;
%% C. LDA
%% 1.a. Load data
data_rubber = load('Processed_data\Data_for_each_contact\oblong_rubber_papillarray_single.mat');
data_TPU = load('Processed_data\Data_for_each_contact\oblong_TPU_papillarray_single.mat');
% Extract tactile displacements (features) and create labels
disp_rubber = data_rubber.tactile_displacements; 
disp_TPU = data_TPU.tactile_displacements;      
disp = [disp_rubber; disp_TPU];           % Combine data (42 x 27)
% Create label
y_rubber = zeros(size(disp_rubber,1), 1); % Label 0 for Rubber
y_TPU = ones(size(disp_TPU,1), 1);        % Label 1 for TPU
y = [y_rubber; y_TPU];                    % Combine labels (42 x 1)
%% 1.b. 3D scatter plot for selected features
disp_sel = disp(:, 13:15);  % Select features D_X, D_Y, D_Z
figure;
scatter3(disp_sel(y == 0, 1), disp_sel(y == 0, 2), disp_sel(y == 0, 3), 50, 'g', 'filled'); % Rubber (Green)
hold on;
scatter3(disp_sel(y == 1, 1), disp_sel(y == 1, 2), disp_sel(y == 1, 3), 50, 'b', 'filled'); % TPU (Blue)
% Labels and title
title('3D Scatter Plot of Tactile Displacements');
xlabel('D_X');
ylabel('D_Y');
zlabel('D_Z');
legend('Rubber', 'TPU');
grid on;
view(40, 30);  % Adjust view angle for better visualization
%% 1.c. LDA for combination of 2D
% Standardize the data
disp = zscore(disp);
disp_sel = disp(:, 13:15);
% Step 1: Calculate class means
classes = unique(y);
mu = zeros(length(classes), size(disp, 2));
for i = 1:length(classes)
    mu(i, :) = mean(disp(y == classes(i), :), 1);
end
% Step 2: Compute Within-Class Scatter Matrix
Sw = zeros(size(disp, 2));
for i = 1:length(classes)
    class_data = disp(y == classes(i), :);
    Si = cov(class_data);
    Sw = Sw + Si;
end
% Step 3: Compute Between-Class Scatter Matrix
mu_overall = mean(disp, 1);
Sb = zeros(size(disp, 2));
for i = 1:length(classes)
    Ni = sum(y == classes(i));
    diff = (mu(i, :) - mu_overall)';
    Sb = Sb + Ni * (diff * diff');
end
% Step 4: Solve eigenvalue problem
[V, D] = eig(pinv(Sw) * Sb);
% Step 5: Sort eigenvalues and select top eigenvectors
[eigenvalues_sorted, idx] = sort(diag(D), 'descend');
W = V(:, idx);
% Step 6: Select the first two components for 2D projection
W_2D = W(:, 1:2);
X_lda_2D = disp * W_2D;  % Project data onto 2D LDA space
figure;
pair_labels = {'D_X', 'D_Y', 'D_Z'};
pair_idx = [1 2; 1 3; 2 3];  % Combinations for (D_X, D_Y), (D_X, D_Z), (D_Y, D_Z)
for i = 1:3
    % Select the feature pair for the current plot
    x_idx = pair_idx(i, 1);
    y_idx = pair_idx(i, 2);
    disp_2d = disp_sel(:, [x_idx, y_idx]);
    
    lda_vector = W([x_idx, y_idx], 1); 
    % Project raw data points onto the LDA line
    proj_scale = (disp_2d - mean(disp_2d, 1)) * lda_vector / (lda_vector' * lda_vector);
    disp_projected = mean(disp_2d, 1) + proj_scale * lda_vector';  % Projected points

    % Project data points to LDA line
    subplot(1, 3, i);
    scatter(disp_sel(y == 0, x_idx), disp_sel(y == 0, y_idx), 50, 'g', 'filled'); hold on;
    scatter(disp_sel(y == 1, x_idx), disp_sel(y == 1, y_idx), 50, 'b', 'filled');
    scatter(disp_projected(y == 0, 1), disp_projected(y == 0, 2), 50, 'g', 'o');  % Rubber projected points
    scatter(disp_projected(y == 1, 1), disp_projected(y == 1, 2), 50, 'b', 'o');  % TPU projected points

    % Re-compute LDA direction in the selected 2D plane
    % W_pair = W_2D([x_idx, y_idx], :);
    % lda_vector = W_pair(:, 1);  % Select the first LDA component

    % Compute and plot a LDA direction line
    x_vals = linspace(min(disp_2d(:, 1)), max(disp_2d(:, 1)), 10);
    y_vals = mean(disp_2d(:, 2)) + (lda_vector(2) / lda_vector(1)) * (x_vals - mean(disp_2d(:, 1)));
    plot(x_vals, y_vals, 'k-', 'LineWidth', 1);

    % Add labels and title
    title(['LDA Projection on ', pair_labels{x_idx}, ' vs ', pair_labels{y_idx}]);
    xlabel(pair_labels{x_idx});
    ylabel(pair_labels{y_idx});
    legend('Rubber', 'TPU', 'LDA Direction');
    grid on;
end
