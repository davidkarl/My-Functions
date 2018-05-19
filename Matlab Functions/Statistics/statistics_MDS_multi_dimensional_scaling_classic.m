function [reduced_data_mat] = statistics_MDS_multi_dimensional_scaling_classic(Z_input_data_mat)
%Reduce to 2 dimensions only for now!!!!

% load cereal
% Z_input_data_mat = cereal;

[number_of_samples,number_of_dimensions] = size(Z_input_data_mat);

%Get the matrix of dissimilarities:
samples_distance_matrix = squareform(pdist(cereal,'cityblock'));

%Now implement the steps for classical MDS:
Q = -0.5*samples_distance_matrix.^2;
H = eye(number_of_samples) - ones(number_of_samples,1)*ones(1,number_of_samples)/number_of_samples;
B = H*Q*H;
[eigen_vectors_mat,eigen_values_mat] = eig(B);
[sorted_eigen_values, sorted_eigen_values_indices] = sort(diag(eigen_values_mat),'descend');
eigen_vectors_mat = eigen_vectors_mat(:,sorted_eigen_values_indices);
%Reduce the dimensionality to 2-D.
X = eigen_vectors_mat(:,1:2)*diag(sqrt(sorted_eigen_values(1:2)));

% % Plot the points and label them.
% plot(X(:,1),X(:,2),'o')
% text(X(:,1),X(:,2),labs)
% title('Classical MDS - City Block Metric')


