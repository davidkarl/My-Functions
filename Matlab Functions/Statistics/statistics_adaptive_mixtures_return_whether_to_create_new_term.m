function flag_create_new_term = statistics_adaptive_mixtures_return_whether_to_create_new_term(...
                                                data_point,...
                                                pies_vec,...
                                                means_vec,...
                                                variances_mat,...
                                                threshold_distance,...
                                                number_of_mixture_terms,...
                                                max_number_of_terms)
% CREATEM   Helper function - multivariate adaptive mixtures.
%  Implements the create rule. 

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox

% compute functional value at data point

flag_create_new_term = 0;
[number_of_dimensions,number_of_clusters] = size(means_vec);

% this rule says that if the point is greater than
% tc sigmas away from each term, then we create:

distance=zeros(1,number_of_mixture_terms);
for i = 1:number_of_mixture_terms
  distance(i)=(data_point'-means_vec(:,i))'*inv(variances_mat(:,(i-1)*number_of_dimensions+1:i*number_of_dimensions))*(data_point'-means_vec(:,i));
end

if min(distance)>threshold_distance & number_of_mixture_terms < max_number_of_terms
	flag_create_new_term=1;
end


