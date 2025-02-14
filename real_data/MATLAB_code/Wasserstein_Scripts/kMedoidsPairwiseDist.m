% Load the Wasserstein distance matrix
% W_dist_matrix = load('wasserstein_distances.mat');
W_dist_matrix = pairwise_distance_matrix;

% Convert struct to array if necessary
if isstruct(W_dist_matrix)
    W_dist_matrix = W_dist_matrix.W_dist_matrix;
end

% Define the number of clusters
k = 3; 

% Run K-Medoids on the distance matrix (no need to specify 'Distance')
[idx, medoids, sumd] = kmedoids(W_dist_matrix, k);

% Display clustering results
disp('Cluster assignments:');
disp(idx);
disp('Medoid indices:');
disp(medoids);

cluster_1 = find(idx == 1)
cluster_2 = find(idx == 2)
cluster_3 = find(idx == 3)

cluster_1_sessions = sesions_for_EMD(cluster_1)
cluster_2_sessions = sesions_for_EMD(cluster_2)
cluster_3_sessions = sesions_for_EMD(cluster_3)