function [cluster_membership_per_sample_vec,number_of_elements_in_each_cluster_vec,centers] = ...
                    statistics_k_means_clustering(Z_input_data_mat,number_of_clusters,cluster_centers_mat)
% CSHMEANS K-means clustering.
%
%   [CID,NR,CENTERS] = CSHMEANS(X,K,NC) Performs K-means
%   clustering using the data given in X. 
%   
%   INPUTS: X is the n x d matrix of data,
%   where each row indicates an observation. K indicates
%   the number of desired clusters. NC is a k x d matrix for the
%   initial cluster centers. If NC is not specified, then the
%   centers will be randomly chosen from the observations.
%
%   OUTPUTS: CID provides a set of n indexes indicating cluster
%   membership for each point. NR is the number of observations
%   in each cluster. CENTERS is a matrix, where each row
%   corresponds to a cluster center.
%
%   See also CSKMEANS


%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox 


warning off
[number_of_samples,number_of_dimensions] = size(Z_input_data_mat);
if nargin < 3
	%Then pick some observations to be the cluster centers:
    closest_cluster_center_index = ceil(number_of_samples*rand(1,number_of_clusters));
	%We will add some noise to make it interesting:
	cluster_centers_mat = Z_input_data_mat(closest_cluster_center_index,:) + randn(number_of_clusters,number_of_dimensions);
end

%Set up storage:
%integer 1,...,k indicating cluster membership:
cluster_membership_per_sample_vec = zeros(1,number_of_samples);	
%Make this different to get the loop started:
cluster_membership_per_sample_vec_previous = ones(1,number_of_samples);
%The number in each cluster:
number_of_elements_in_each_cluster_vec = zeros(1,number_of_clusters);	
%Set up maximum number of iterations:
max_number_of_iterations = 100;
iteration_counter = 1;

while ~isequal(cluster_membership_per_sample_vec,cluster_membership_per_sample_vec_previous) && iteration_counter < max_number_of_iterations
    %Assign previous cluster membership as current:
    cluster_membership_per_sample_vec_previous = cluster_membership_per_sample_vec;
	
    %For each point, find the distance to all cluster centers:
	for i = 1:number_of_samples
		distance_between_current_sample_and_cluster_centers = ...
                    sum((repmat(Z_input_data_mat(i,:),number_of_clusters,1)-cluster_centers_mat).^2,2);
		[m,closest_cluster_center_index] = min(distance_between_current_sample_and_cluster_centers);	% assign it to this cluster center
		cluster_membership_per_sample_vec(i) = closest_cluster_center_index;
    end
    
	%Find the new cluster centers:
	for i = 1:number_of_clusters
		%find all points in current cluster:
		samples_belonging_to_current_cluster_indices = find(cluster_membership_per_sample_vec==i);
		%find the centroid:
		cluster_centers_mat(i,:) = mean(Z_input_data_mat(samples_belonging_to_current_cluster_indices,:));
		%Find the number in each cluster:
		number_of_elements_in_each_cluster_vec(i) = length(samples_belonging_to_current_cluster_indices);
    end
    
    %Uptick iteration counter:
	iteration_counter = iteration_counter + 1;
end
centers = cluster_centers_mat;
warning on





