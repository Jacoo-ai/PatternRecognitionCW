%% Step 1: Load Data and Apply PCA
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


%% Step 2: Find Optimal Number of Trees
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
ylabel('Mean OOB Error (Cross-Validation)');
title('Out-of-Bag Error vs Number of Trees (Cross-Validation)');
grid on;

hold on;
plot(stable_tree_idx, mean_oob_error(stable_tree_idx), 'go', 'MarkerSize', 8, 'LineWidth', 2);
legend('Mean OOB Error', 'Stable Number of Trees');




%% Step 3: Split Data into Train and Test Sets (60% Train, 40% Test)
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


%% Step 4: Train the Model and Visualize 2 Trees

% Train the Random Forest model using the optimal number of trees
num_trees = stable_tree_idx;
rf_model = TreeBagger(num_trees, train_data, train_labels, 'OOBPrediction', 'on');

% Extract two decision trees from the trained Random Forest model
tree1 = rf_model.Trees{1};  % First tree
tree2 = rf_model.Trees{2};  % Second tree

% Visualize the first decision tree
figure;
view(tree1, 'Mode', 'graph'); % 'graph' mode shows a tree diagram
title('Decision Tree 1');

% Visualize the second decision tree
figure;
view(tree2, 'Mode', 'graph'); % 'graph' mode shows a tree diagram
title('Decision Tree 2');


%% Step 5: Display Confusion Matrix

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
