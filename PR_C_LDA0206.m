clc; clear; close all;
%C. LDA
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
scatter3(disp_sel(y == 0, 1), disp_sel(y == 0, 2), disp_sel(y == 0, 3), 20, 'g', 'filled'); % Rubber (Green)
hold on;
scatter3(disp_sel(y == 1, 1), disp_sel(y == 1, 2), disp_sel(y == 1, 3), 20, 'b', 'filled'); % TPU (Blue)
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
% X_lda_2D = disp * W_2D;  % Project data onto 2D LDA space
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
    subplot('Position', [0.05 + (i - 1) * 0.3, 0.6, 0.27, 0.35]); 
    scatter(disp_sel(y == 0, x_idx), disp_sel(y == 0, y_idx), 20, 'g', 'filled', 'DisplayName', 'Rubber'); hold on;
    scatter(disp_sel(y == 1, x_idx), disp_sel(y == 1, y_idx), 20, 'b', 'filled', 'DisplayName', 'TPU');
    scatter(disp_projected(y == 0, 1), disp_projected(y == 0, 2), 20, 'g', 'o', 'DisplayName', 'Projected Rubber');  % Rubber projected points
    scatter(disp_projected(y == 1, 1), disp_projected(y == 1, 2), 20, 'b', 'o', 'DisplayName', 'Projected TPU');  % TPU projected points

    % Re-compute LDA direction in the selected 2D plane
    % W_pair = W_2D([x_idx, y_idx], :);
    % lda_vector = W_pair(:, 1);  % Select the first LDA component

    % Compute and plot a LDA direction line
    xlim([-4, 4]); 
    ylim([-4, 4]);
    x_range = xlim;
    y_range = ylim;
    x_vals = linspace(x_range(1), x_range(2), 100);
    y_vals = mean(disp_2d(:, 2)) + (lda_vector(2) / lda_vector(1)) * (x_vals - mean(disp_2d(:, 1)));
    plot(x_vals, y_vals, 'k-', 'LineWidth', 1.0, 'DisplayName', 'LDA Direction');
    
    center_point = mean(disp_projected, 1);  % 投影点的中心
    discrimination_slope = -lda_vector(1) / lda_vector(2);  % 法向量斜率（垂直于 LDA）
    % 计算判别边界的坐标
    x_decision_vals = linspace(min(disp_2d(:, 1)), max(disp_2d(:, 1)), 100);
    y_decision_vals = center_point(2) + discrimination_slope * (x_decision_vals - center_point(1));
    % 绘制判别边界
    plot(x_decision_vals, y_decision_vals, 'k--', 'LineWidth', 1.0, 'DisplayName', 'Discrimination Line');

    % Add labels and title
    title(['LDA Projection on ', pair_labels{x_idx}, ' vs ', pair_labels{y_idx}]);
    xlabel(pair_labels{x_idx});
    ylabel(pair_labels{y_idx});
    % xlim([min(disp_2d(:, 1)), max(disp_2d(:, 1))]);
    % ylim([min(disp_2d(:, 2)), max(disp_2d(:, 2))]);
    legend('Location', 'SouthWest');
    grid on;

    subplot('Position', [0.05 + (i - 1) * 0.3, 0.35, 0.27, 0.10]);
    scatter(proj_scale(y == 0), zeros(sum(y == 0), 1), 20, 'g', 'filled'); hold on;
    scatter(proj_scale(y == 1), zeros(sum(y == 1), 1), 20, 'b', 'filled');

    % Add labels and title
    title('1D Projection onto LDA');
    xlabel('LD1');
    ylabel('');
    % xlim([min(proj_scale), max(proj_scale)]);  % Synchronize x-axis with 1D projection range
    yticks([]);  % Remove y-axis ticks for a cleaner look
    grid on;
end
%% 1.d. LDA for 3D data
% Select the top 3 eigenvectors for 3D LDA projection
W_3D = W(:, 1:3);
figure;
hold on;
disp_sel = zscore(disp_sel);
% 绘制原始数据点
scatter3(disp_sel(y == 0, 1), disp_sel(y == 0, 2), disp_sel(y == 0, 3), 20, 'g', 'filled', 'DisplayName', 'Rubber');
scatter3(disp_sel(y == 1, 1), disp_sel(y == 1, 2), disp_sel(y == 1, 3), 20, 'b', 'filled', 'DisplayName', 'TPU');

% 绘制 LD1
ld1_center = mean(disp_sel, 1)';  % Center around the mean
ld1_vector = W_3D(13:15, 1);      % LD1 vector for selected features
ld1_start = ld1_center - 10 * ld1_vector;  % Extend start point
ld1_end = ld1_center + 10 * ld1_vector;    % Extend end point
plot3([ld1_start(1), ld1_end(1)], [ld1_start(2), ld1_end(2)], [ld1_start(3), ld1_end(3)], 'k-', 'LineWidth', 2, 'DisplayName', 'LD1');

% 绘制 LD2
ld2_vector = W_3D(13:15, 2);      % LD2 vector for selected features
ld2_start = ld1_center - 10 * ld2_vector;  % Extend start point
ld2_end = ld1_center + 10 * ld2_vector;    % Extend end point
plot3([ld2_start(1), ld2_end(1)], [ld2_start(2), ld2_end(2)], [ld2_start(3), ld2_end(3)], 'm-', 'LineWidth', 2, 'DisplayName', 'LD2');
% Generate the plane
% Define grid for plane in LD1 and LD2 directions
t_range = linspace(-10, 10, 50);  % Expand the range to match LD line length
[s, t] = meshgrid(t_range, t_range);

% Compute plane points
X_plane = ld1_center(1) + s * ld1_vector(1) + t * ld2_vector(1);
Y_plane = ld1_center(2) + s * ld1_vector(2) + t * ld2_vector(2);
Z_plane = ld1_center(3) + s * ld1_vector(3) + t * ld2_vector(3);

% Plot the plane
surf(X_plane, Y_plane, Z_plane, 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'DisplayName', 'LD1-LD2 Plane');

% Set plot properties
xlabel('D_X');
ylabel('D_Y');
zlabel('D_Z');
title('LD1 and LD2 with Hyperplane in Original Feature Space');
legend('Location', 'best');
grid on;
view(40, 30);  % Adjust the viewing angle

% Project data onto 2D plane defined by LD1 and LD2
projected_data_2d = [disp_sel * ld1_vector, disp_sel * ld2_vector];  % Projection onto 2D plane

% Plot 2D Projection
figure;
subplot(2, 1, 1);
scatter(projected_data_2d(y == 0, 1), projected_data_2d(y == 0, 2), 20, 'g', 'filled'); hold on;
scatter(projected_data_2d(y == 1, 1), projected_data_2d(y == 1, 2), 20, 'b', 'filled');
xline(0, '--r', 'LineWidth', 1.5);
xlabel('LD1'); ylabel('LD2');
title('2D Projection onto LD1 and LD2 Plane');
grid on;

% 1D Projection along LD1
subplot(2, 1, 2);
ld1_projection = projected_data_2d(:, 1);
scatter(ld1_projection(y == 0), zeros(sum(y == 0), 1), 20, 'g', 'filled'); hold on;
scatter(ld1_projection(y == 1), zeros(sum(y == 1), 1), 20, 'b', 'filled');
xlabel('LD1'); yticks([]);
title('1D Projection onto LD1');
grid on;

% Add class separation line
center_ld1 = mean(ld1_projection);  % Center position for separation
xline(center_ld1, '--r', 'LineWidth', 1.5);  % Plot separation line