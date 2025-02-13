clc; clear; close all;
%% D. Clustering & Classification
%% 
% D.1. Clustering
% Choose hexagon objects for clustering task
% 1.a. Plot the raw data.
NORMAL = load('Processed_data\Data_for_each_contact\hexagon_papillarray_single.mat');
RUBBER = load('Processed_data\Data_for_each_contact\hexagon_rubber_papillarray_single.mat');
TPU = load('Processed_data\Data_for_each_contact\hexagon_TPU_papillarray_single.mat');
% Extract tactile displacements (features) and create labels
dspm_NORMAL = NORMAL.tactile_displacements;
dspm_RUBBER = RUBBER.tactile_displacements;
dspm_TPU = TPU.tactile_displacements;
dspm = [dspm_NORMAL; dspm_RUBBER; dspm_TPU]; 
dspm_P4 = dspm(:, 13:15);
% Create label
% Generate scatter plot for different materials
figure;
scatter3(dspm_NORMAL(:, 13), dspm_NORMAL(:, 14), dspm_NORMAL(:, 15), 30, 'r', 'filled'); hold on;
scatter3(dspm_RUBBER(:, 13), dspm_RUBBER(:, 14), dspm_RUBBER(:, 15), 30, 'g', 'filled');
scatter3(dspm_TPU(:, 13), dspm_TPU(:, 14), dspm_TPU(:, 15), 30, 'b', 'filled');
xlabel('X-displacement'); ylabel('Y-displacement'); zlabel('Z-displacement');
% title('Scatter plot of tactile displacements for different materials');
legend('Normal', 'Rubber', 'TPU');
grid on; box on; hold off;



% 1.b. Apply k-means clustering
cluster_colors = [0.8500 0.3250 0.0980;  % Cluster 1 - Red
                  0.4660 0.6740 0.1880;  % Cluster 2 - Green
                  0 0.4470 0.7410];      % Cluster 3 - Blue
k = 3;  % Number of clusters corresponding to the three materials
[cluster_indices, cluster_centers] = kmeans(dspm_P4, k);

% Plot clustering result
figure; hold on;
% Assign different colors to each cluster for better visualization
for i = 1:k
    scatter3(dspm_P4(cluster_indices == i, 1), dspm_P4(cluster_indices == i, 2), dspm_P4(cluster_indices == i, 3), ...
             30, cluster_colors(i, :), 'filled', 'DisplayName', sprintf('Cluster %d', i));
    scatter3(cluster_centers(i, 1), cluster_centers(i, 2), cluster_centers(i, 3), ...
             200, cluster_colors(i, :), 'p', 'filled', 'DisplayName', sprintf('Centroid %d', i), 'MarkerEdgeColor', 'k', 'LineWidth', 1.0); 
end
hold off;

% Add labels and title
xlabel('X-displacement'); ylabel('Y-displacement'); zlabel('Z-displacement');

% Add legend for each cluster
legend('show', 'Location', 'best');

% Adjust view and grid for better 3D effect
view(3);  % Ensure the view is set to 3D
grid on; box on;



% 1.c. Apply k-means with a different distance metric (e.g., cityblock)
% L1 distance

% --- Helper function: Map cluster labels to match Euclidean clustering ---
function [mapped_indices, sorted_centers] = map_clusters(original_centers, new_centers, indices)
    % Calculate pairwise distances between original and new centers
    distances = pdist2(original_centers, new_centers);
    % Find the best matching order of clusters
    [~, map] = min(distances, [], 2);
    % Apply the mapping to cluster indices
    mapped_indices = arrayfun(@(x) find(map == x), indices);
    % Apply the same mapping to cluster centers
    sorted_centers = new_centers(map, :);
end
% --- function end ---

[cluster_indices_L1, cluster_centers_L1] = kmeans(dspm_P4, k, 'Distance', 'cityblock');
[cluster_indices_L1, cluster_centers_L1] = map_clusters(cluster_centers, cluster_centers_L1, cluster_indices_L1);
% Plot clustering result with new distance metric
figure; hold on;
% Assign different colors to each cluster for better visualization
for i = 1:k
    scatter3(dspm_P4(cluster_indices_L1 == i, 1), dspm_P4(cluster_indices_L1 == i, 2), dspm_P4(cluster_indices_L1 == i, 3), ...
             30, cluster_colors(i, :), 'filled', 'Displayname', sprintf('Cluster %d', i));
    scatter3(cluster_centers_L1(i, 1), cluster_centers_L1(i, 2), cluster_centers_L1(i, 3), ...
             200, cluster_colors(i, :), 'p', 'filled', 'DisplayName', sprintf('Centroid %d', i), 'MarkerEdgeColor', 'k', 'LineWidth', 1.0);
end

hold off;

% Add labels and title
xlabel('X-displacement'); ylabel('Y-displacement'); zlabel('Z-displacement');

% Add legend for each cluster
legend('show', 'Location', 'best');

% Adjust view and grid for better 3D effect
view(3);  % Ensure the view is set to 3D
grid on; box on;



%% D.2. Classification
% Step 1: Load Data and Apply PCA
% Define file paths for tactile displacement data
mat_files = {
    "Processed_data\Data_for_each_contact\cylinder_papillarray_single.mat", 
    "Processed_data\Data_for_each_contact\cylinder_rubber_papillarray_single.mat", 
    "Processed_data\Data_for_each_contact\cylinder_TPU_papillarray_single.mat"
};

all_displacements = [];
labels = [];

% Load and concatenate displacement data from multiple files
for i = 1:length(mat_files)
    data = load(mat_files{i});
    displacements = data.tactile_displacements;
    all_displacements = [all_displacements; displacements];
    labels = [labels; i * ones(size(displacements, 1), 1)];
end

% Standardize the displacement data
standardized_displacements = (all_displacements - mean(all_displacements)) ./ std(all_displacements);

% Apply PCA and reduce to the first two principal components
[~, score] = pca(standardized_displacements);
reduced_data = score(:, 1:2);  % Select first two principal components


% Step 2: Find Optimal Number of Trees
K = 10;
num_trees = 100;  % Maximum number of trees
% Store OOB errors for all folds
oob_errors_all = zeros(K, num_trees);

for k = 1:K
    fprintf('Trial %d/%d...\n', k, K);
    num_samples = size(reduced_data, 1);
    num_train = round(0.6 * num_samples);
    % Randomly shuffle indices
    rand_indices = randperm(num_samples);
    train_indices = rand_indices(1:num_train); 
    train_data = reduced_data(train_indices, :);
    train_labels = labels(train_indices);
    % Train a Random Forest model with Out-of-Bag (OOB) estimation
    rf_model = TreeBagger(num_trees, train_data, train_labels, 'OOBPrediction', 'on');
    % Store OOB error for this trial
    oob_errors_all(k, :) = oobError(rf_model);
end

% Compute mean OOB error across trials
mean_oob_error = mean(oob_errors_all, 1);

% Find the first tree count where OOB error stabilizes
error_diff = abs(diff(mean_oob_error)); % Compute adjacent error differences
convergence_threshold = 0.0005; 
stable_tree_idx = find(error_diff < convergence_threshold, 1);

% If no stable point found, use the minimum OOB error point
if isempty(stable_tree_idx)
    stable_tree_idx = final_tree_count;
end

fprintf('Optimal number of trees after convergence: %d\n', stable_tree_idx);

% Plot OOB error curve
figure;
plot(1:num_trees, mean_oob_error, '-o', 'LineWidth', 1.5);
xlabel('Number of Trees');
ylabel('Mean OOB Error');
title('Out-of-Bag Error vs Number of Trees');
grid on; box on;

hold on;
plot(stable_tree_idx, mean_oob_error(stable_tree_idx), 'go', 'MarkerSize', 8, 'LineWidth', 2);
legend('Mean OOB Error', 'Optimal Number of Trees');


% Step 3: Split Data into Train and Test Sets (60% Train, 40% Test)
num_samples = size(reduced_data, 1);
num_train = round(0.6 * num_samples);

% Randomly shuffle indices
rand_indices = randperm(num_samples);

% Create Train and Test Sets
train_indices = rand_indices(1:num_train);
test_indices = rand_indices(num_train+1:end);

train_data = reduced_data(train_indices, :);
train_labels = labels(train_indices);

test_data = reduced_data(test_indices, :);
test_labels = labels(test_indices);


% Step 4: Train the Model and Visualize 2 Trees
% Train the Random Forest model using the optimal number of trees
num_trees = stable_tree_idx;
rf_model = TreeBagger(num_trees, train_data, train_labels, 'OOBPrediction', 'on');

% Extract two decision trees from the trained Random Forest model
tree1 = rf_model.Trees{1};  % First tree
tree2 = rf_model.Trees{2};  % Second tree

% Visualize the first decision tree
view(tree1, 'Mode', 'graph'); % 'graph' mode shows a tree diagram

% Visualize the second decision tree
view(tree2, 'Mode', 'graph'); % 'graph' mode shows a tree diagram


% Step 5: Display Confusion Matrix

% Predict test labels using the trained Random Forest model
predicted_labels = str2double(predict(rf_model, test_data)); % Convert to numeric labels

% Generate confusion matrix
conf_matrix = confusionmat(test_labels, predicted_labels);
disp('Confusion Matrix:');
disp(conf_matrix);

% Plot confusion matrix as a heatmap
figure;
heatmap(conf_matrix, 'Colormap', parula);
xlabel('Predicted Class');
ylabel('Actual Class');
title('Confusion Matrix for Random Forest Model');

% Compute classification accuracy
accuracy = sum(diag(conf_matrix)) / sum(conf_matrix(:));
fprintf('Overall Accuracy: %.2f%%\n', accuracy * 100);