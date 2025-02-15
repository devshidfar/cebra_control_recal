% Load the Wasserstein distance matrix
% W_dist_matrix = load('wasserstein_distances.mat');
%% Given: your pairwise Wasserstein distance matrix
W_dist_matrix = pairwise_EMD_distance_matrix;  % 15x15 matrix

%% Step 1: Convert distance matrix to similarity matrix
% Choose sigma. One common choice is the median of the distances.
sigma = median(W_dist_matrix(:));
S = exp( - (W_dist_matrix.^2) / (2*sigma^2) );  % Similarity matrix

%% Step 2: Construct the normalized graph Laplacian
% Compute the degree matrix
D = diag(sum(S, 2));

% Compute D^(-1/2)
D_inv_sqrt = diag(1 ./ sqrt(diag(D)));

% Compute the symmetric normalized Laplacian: L_sym = I - D^(-1/2) * S * D^(-1/2)
L = eye(size(S)) - D_inv_sqrt * S * D_inv_sqrt;

%% Step 3: Compute the eigenvectors of the Laplacian
% Since the matrix is small, we can use eig. Sort the eigenvalues in ascending order.
[V, eigenVals] = eig(L);
[eigSorted, idx] = sort(diag(eigenVals));
V = V(:, idx);

%% Step 4: Select the first k eigenvectors and normalize rows
k = 2;  % Set desired number of clusters
U = V(:, 1:k);

% Normalize each row of U to have unit length
rowNorms = sqrt(sum(U.^2, 2));
U_normalized = bsxfun(@rdivide, U, rowNorms);

%% Step 5: Run k-means clustering on the rows of U_normalized
opts = statset('Display','final');
clusterLabels = kmeans(U_normalized, k, 'Replicates', 100, 'Options', opts);

%% Display the cluster assignments
disp('Cluster labels for each point cloud:');
disp(clusterLabels);


% Get the number of point clouds
N = size(W_dist_matrix, 1);

% Create an upper triangular mask (excluding diagonal)
mask = triu(true(N), 1);

% Apply the mask (set lower-triangle elements to NaN)
W_upper = W_dist_matrix;
W_upper(~mask) = NaN; 

% Plot the upper triangular Wasserstein distance matrix
imagesc(W_upper);
colorbar;
title('Upper Triangle of Pairwise Wasserstein Distance Matrix');
xlabel('Point Clouds');
ylabel('Point Clouds');


cluster_1 = find(idx == 1)
cluster_2 = find(idx == 2)
cluster_3 = find(idx == 3)

cluster_1_sessions = sesions_for_EMD(cluster_1)
cluster_2_sessions = sesions_for_EMD(cluster_2)
cluster_3_sessions = sesions_for_EMD(cluster_3)