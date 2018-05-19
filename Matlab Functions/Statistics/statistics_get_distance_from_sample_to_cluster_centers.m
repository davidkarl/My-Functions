function dist = statistics_get_distance_from_sample_to_cluster_centers(x,cluster_centers_vec)
% This returns a vector of k distances from observation x to all of the
% cluster centers.
[number_of_clusters,number_of_dimensions] = size(cluster_centers_vec);
dist = sum((repmat(x,number_of_clusters,1)-cluster_centers_vec).^2,2);
