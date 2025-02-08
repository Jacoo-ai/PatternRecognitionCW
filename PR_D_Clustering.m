clc; clear; close all;
%% D. Clustering & Classification
%% 
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
grid on; hold off;
%%
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
             30, cluster_colors(i, :), 'filled');
end
hold off;

% Add labels and title
xlabel('X-displacement'); ylabel('Y-displacement'); zlabel('Z-displacement');
title('K-means clustering with Euclidean distance metric');

% Add legend for each cluster
legend('Cluster 1', 'Cluster 2', 'Cluster 3');

% Adjust view and grid for better 3D effect
view(3);  % Ensure the view is set to 3D
grid on;
%%
% 1.c. Apply k-means with a different distance metric (e.g., cityblock)
% L1 distance

% --- Helper function: Map cluster labels to match Euclidean clustering ---
function mapped_indices = map_clusters(original_centers, new_centers, indices)
    % Calculate pairwise distances between original and new centers
    distances = pdist2(original_centers, new_centers);
    % Find the best matching order of clusters
    [~, map] = min(distances, [], 2);
    % Apply the mapping to cluster indices
    mapped_indices = arrayfun(@(x) find(map == x), indices);
end
% --- function end ---

[cluster_indices_L1, cluster_centers_L1] = kmeans(dspm_P4, k, 'Distance', 'cityblock');
cluster_indices_L1 = map_clusters(cluster_centers, cluster_centers_L1, cluster_indices_L1);
% Plot clustering result with new distance metric
figure; hold on;
% Assign different colors to each cluster for better visualization
for i = 1:k
    scatter3(dspm_P4(cluster_indices_L1 == i, 1), dspm_P4(cluster_indices_L1 == i, 2), dspm_P4(cluster_indices_L1 == i, 3), ...
             30, cluster_colors(i, :), 'filled');
end
hold off;

% Add labels and title
xlabel('X-displacement'); ylabel('Y-displacement'); zlabel('Z-displacement');
title('K-means clustering with cityblock distance metric');

% Add legend for each cluster
legend('Cluster 1', 'Cluster 2', 'Cluster 3');

% Adjust view and grid for better 3D effect
view(3);  % Ensure the view is set to 3D
grid on;