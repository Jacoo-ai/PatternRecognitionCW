clear; clc; close all;
%% C. LDA
% Load data
data_rubber = load('Processed_data\Data_for_each_contact\oblong_rubber_papillarray_single.mat');
data_TPU = load('Processed_data\Data_for_each_contact\oblong_TPU_papillarray_single.mat');
% Extract tactile displacements (features) and create labels
disp_rubber = data_rubber.tactile_displacements'; 
disp_TPU = data_TPU.tactile_displacements';      
disp = [disp_rubber; disp_TPU];           % Combine data (42 x 27)

y_rubber = zeros(size(disp_rubber,1), 1); % Label 0 for Rubber
y_TPU = ones(size(disp_TPU,1), 1);        % Label 1 for TPU
y = [y_rubber; y_TPU];                    % Combine labels (42 x 1)

%% 3D scatter plot
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
%%
% Apply LDA
lda = fitcdiscr(disp, y);

% Project data to 2D
disp_lda_2D = disp * lda.Coeffs(1,2).Linear; % Use LDA coefficients

% Plot 2D LDA projection
figure;
scatter(disp_lda_2D(y==0,1), zeros(sum(y==0),1), 50, 'g', 'filled'); % Rubber (Green)
hold on;
scatter(disp_lda_2D(y==1,1), zeros(sum(y==1),1), 50, 'b', 'filled'); % TPU (Blue)
title('LDA Projection (2D)');
xlabel('LDA Component 1');
ylabel('Zero (for visualization)');
legend('Rubber', 'TPU');
grid on;

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
