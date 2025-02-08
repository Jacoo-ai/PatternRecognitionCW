clear; clc; close all;
%% C. LDA
% 1.a. Load data
data_rubber = load('Processed_data\Data_for_each_contact\oblong_rubber_papillarray_single.mat');
data_TPU = load('Processed_data\Data_for_each_contact\oblong_TPU_papillarray_single.mat');
% Extract tactile displacements (features) and create labels
disp_rubber = data_rubber.tactile_displacements'; 
disp_TPU = data_TPU.tactile_displacements';      
disp = [disp_rubber; disp_TPU];           % Combine data (42 x 27)

y_rubber = zeros(size(disp_rubber,1), 1); % Label 0 for Rubber
y_TPU = ones(size(disp_TPU,1), 1);        % Label 1 for TPU
y = [y_rubber; y_TPU];                    % Combine labels (42 x 1)

%% 1.b. 3D scatter plot
% Standardize the data
disp = zscore(disp);
disp_sel = disp(:, 13:15);
% Plot 3D scatter
figure;
scatter3(disp_sel(y==0,1), disp_sel(y==0,2), disp_sel(y==0,3), 50, 'g', 'filled'); % Rubber (Green)
hold on;
scatter3(disp_sel(y==1,1), disp_sel(y==1,2), disp_sel(y==1,3), 50, 'b', 'filled'); % TPU (Blue)

% Labels and title
title('3D Scatter Plot of Tactile Displacements');
xlabel('D_X');
ylabel('D_Y');
zlabel('D_Z');
legend('Rubber', 'TPU');
grid on;
view(40, 30); % Adjust view angle for better visualization
%% 1.c.
% Apply LDA
lda = fitcdiscr(disp, y); % Train LDA model
disp_lda_2D = disp * lda.Coeffs(1,2).Linear; % Project data to 2D using LDA
% Plot 2D combinations with LDA direction
% Define feature pairs and labels
pair_labels = {'D_X', 'D_Y', 'D_Z'};
pair_idx = [1 2; 1 3; 2 3]; % (X, Y), (X, Z), (Y, Z)

figure;
for i = 1:3
    % Select feature indices for the current 2D combination
    x_idx = pair_idx(i,1);
    y_idx = pair_idx(i,2);

    disp_2d = disp_sel(:, [x_idx, y_idx]);

    % Train LDA on the selected 2D features
    lda_2d = fitcdiscr(disp_2d, y);
    lda_vector = lda_2d.Coeffs(1, 2).Linear;  % Get LDA vector
    
    % Plot raw data points for both classes
    subplot(1,3,i);
    scatter(disp_sel(y==0, x_idx), disp_sel(y==0, y_idx), 50, 'g', 'filled'); hold on;
    scatter(disp_sel(y==1, x_idx), disp_sel(y==1, y_idx), 50, 'b', 'filled');
    
    % Compute LDA direction line 
    x_vals = linspace(min(disp_2d(:, 1)), max(disp_2d(:, 1)), 100);
    y_vals = mean(disp_2d(:, 2)) + (lda_vector(2) / lda_vector(1)) * (x_vals - mean(disp_2d(:, 1)));
    plot(x_vals, y_vals, 'k-', 'LineWidth', 1);

    % Project data points onto LDA line
    projection_scale = (disp_2d - mean(disp_2d, 1)) * lda_vector / (lda_vector' * lda_vector); % Scaling factor for projection
    disp_projected = mean(disp_2d, 1) + projection_scale * lda_vector'; % Calculate projection points

    % Plot the projection points
    scatter(disp_projected(y==0, 1), disp_projected(y==0, 2), 30, 'g'); % Rubber projected points
    scatter(disp_projected(y==1, 1), disp_projected(y==1, 2), 30, 'b'); % TPU projected points

    % Add plot title, labels, and legend
    title(['LDA Projection on ', pair_labels{x_idx}, ' vs ', pair_labels{y_idx}]);
    xlabel(pair_labels{x_idx});
    ylabel(pair_labels{y_idx});
    legend('Rubber', 'TPU', 'LDA Direction');
    grid on;
end
%%
% Apply LDA to reduce to 3D
[W, ~] = lda_transform(disp, y, 3); % Custom function to compute 3D LDA projection
X_lda_3D = disp * W; % Project data to 3D

% Plot 3D LDA projection
figure;
scatter3(X_lda_3D(y==0,1), X_lda_3D(y==0,2), X_lda_3D(y==0,3), 'g', 'filled'); % Rubber
hold on;
scatter3(X_lda_3D(y==1,1), X_lda_3D(y==1,2), X_lda_3D(y==1,3), 'b', 'filled'); % TPU
title('LDA Projection (3D)');
xlabel('LDA Component 1');
ylabel('LDA Component 2');
zlabel('LDA Component 3');
legend('Rubber', 'TPU');
grid on;

% Plot decision plane
[X1, X2] = meshgrid(linspace(min(X_lda_3D(:,1)), max(X_lda_3D(:,1)), 10), ...
                     linspace(min(X_lda_3D(:,2)), max(X_lda_3D(:,2)), 10));
X3 = (-lda.Coeffs(1,2).Const - lda.Coeffs(1,2).Linear(1) * X1 - lda.Coeffs(1,2).Linear(2) * X2) / lda.Coeffs(1,2).Linear(3);
mesh(X1, X2, X3, 'FaceAlpha', 0.5, 'EdgeColor', 'k');

function [W, mu] = lda_transform(X, y, k)
    % Compute mean vectors
    classes = unique(y);
    mu = zeros(length(classes), size(X,2));
    for i = 1:length(classes)
        mu(i,:) = mean(X(y==classes(i),:),1);
    end

    % Compute within-class scatter matrix
    Sw = zeros(size(X,2));
    for i = 1:length(classes)
        class_data = X(y==classes(i),:);
        Si = cov(class_data);
        Sw = Sw + Si;
    end

    % Compute between-class scatter matrix
    mu_overall = mean(X,1);
    Sb = zeros(size(X,2));
    for i = 1:length(classes)
        Ni = sum(y==classes(i));
        diff = (mu(i,:) - mu_overall)';
        Sb = Sb + Ni * (diff * diff');
    end

    % Solve eigenvalue problem
    [V, D] = eig(pinv(Sw) * Sb);
    [~, idx] = sort(diag(D), 'descend');
    W = V(:, idx(1:k));
end
