function [mixture_pies_vec,mixture_means_mat,mixture_covariance_mat] = statistics_adaptive_mixtures_EM(Z_input_data_mat,...
                                                           max_number_of_mixture_terms,...
                                                           new_cluster_creation_threshold)
% CSADPMIX  Adaptive mixtures density estimation.
%
%   [WGTS,MU,COVM] = CSADPMIX(X,MAXTERMS,TC)
%   This function returns the adaptive mixtures density
%   estimate based on the observations in X.
%
%   X is the data matrix, where each row is an observation.
%   MAXTERMS is the maximum allowed number of terms.
%   TC is the create threshold. This is optional. Default
%   values are available for 1-D, 2-D and 3-D data.
%
%   WGTS is a vector of weights for each component density.
%   MU is an array of means, where each column corresponds to a component mean.
%   COVM is a 3-D array of covariance matrices. Each page (third dimension)
%   is a component d x d covariance matrix. In the univariate case, this is a
%   vector of component variances.
%
%   EXAMPLE
%
%   load snowfall
%   [wgts, mus, vars] = csadpmix(snowfall,25);

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox

[number_of_samples,number_of_dimensions] = size(Z_input_data_mat);

if number_of_samples==1 || number_of_dimensions==1
    % then it is univariate data.
    Z_input_data_mat = Z_input_data_mat(:);
    number_of_dimensions=1;
end

if nargin == 2
    if number_of_dimensions == 1
        new_cluster_creation_threshold = 1;
    elseif number_of_dimensions == 2
        new_cluster_creation_threshold = 2.34;
    elseif number_of_dimensions == 3
        new_cluster_creation_threshold = 3.54;
    else
        error('You must specify the create threshold.')
    end
end

if number_of_dimensions == 1
    %Do the univariate case:
    number_of_samples = length(Z_input_data_mat);
    mixture_means_mat = zeros(1,max_number_of_mixture_terms);
    mixture_variances_mat = zeros(1,max_number_of_mixture_terms);
    mixture_pies_vec = zeros(1,max_number_of_mixture_terms); 
    posterior = zeros(1,max_number_of_mixture_terms);
    %lower bound on new pies:
    min_bound_on_new_pie = 0.00001;					
    %bound on variance:
    maximum_bound_on_variance = 1000;			
    %initialze density to first data point:
    number_of_mixture_terms = 1;
    mixture_means_mat(1) = Z_input_data_mat(1);
    %rule of thumb for initial variance - univariate:
    mixture_variances_mat(1)=(std(Z_input_data_mat))^2/2.5;							
    mixture_pies_vec(1)=1;
    %loop through all of the data points:
    for sample_counter = 2:number_of_samples
        %Calculate mahalanobis distance:
        mahalanobis_distance = ...
            ((Z_input_data_mat(sample_counter)-mixture_means_mat(1:number_of_mixture_terms)).^2)./mixture_variances_mat(1:number_of_mixture_terms);
        
        %Check if distance is large enough to warrant the creation of another mixutre term:
        if min(mahalanobis_distance)>new_cluster_creation_threshold && number_of_mixture_terms<max_number_of_mixture_terms
            flag_create_new_term = 1;
        else
            flag_create_new_term = 0;
        end
        
        %Update or create another term:
        if flag_create_new_term == 0						
            %update terms:
            posterior(1:number_of_mixture_terms) = ...
                statistics_adaptive_mixtures_get_posterior_probabilities(Z_input_data_mat(sample_counter),mixture_pies_vec,mixture_means_mat,mixture_variances_mat,number_of_mixture_terms);
            
            [mixture_pies_vec(1:number_of_mixture_terms),...
             mixture_means_mat(1:number_of_mixture_terms),...
             mixture_variances_mat(1:number_of_mixture_terms)] = ...
                            statistics_adaptive_mixtures_update_parameters(...
                                    Z_input_data_mat(sample_counter),mixture_pies_vec,mixture_means_mat,...
                                    mixture_variances_mat,posterior,number_of_mixture_terms,sample_counter);
        else
            %create a new term:
            number_of_mixture_terms = number_of_mixture_terms+1;
            mixture_means_mat(number_of_mixture_terms) = Z_input_data_mat(sample_counter);
            mixture_pies_vec(number_of_mixture_terms) = max([1/(sample_counter),min_bound_on_new_pie]);											
            %update pies:
            mixture_pies_vec(1:number_of_mixture_terms-1)=...
                mixture_pies_vec(1:number_of_mixture_terms-1)*(1-mixture_pies_vec(number_of_mixture_terms));
            mixture_variances_mat(number_of_mixture_terms)=...
                statistics_adaptive_mixtures_set_new_term_variance(...
                                mixture_means_mat, ...
                                mixture_pies_vec, ...
                                mixture_variances_mat, ...
                                Z_input_data_mat(sample_counter), ...
                                number_of_mixture_terms-1);
        end %end if (create new term)
        %to prevent spiking of variances:
        index = find(mixture_variances_mat(1:number_of_mixture_terms) < 1/(maximum_bound_on_variance*number_of_mixture_terms));
        mixture_variances_mat(index) = ones(size(index))/(maximum_bound_on_variance*number_of_mixture_terms);
    end    % for i loop
    
    %clean up the model - get rid of the 0 terms:
    mixture_means_mat((number_of_mixture_terms+1):max_number_of_mixture_terms) = [];
    mixture_pies_vec((number_of_mixture_terms+1):max_number_of_mixture_terms) = [];
    mixture_variances_mat((number_of_mixture_terms+1):max_number_of_mixture_terms) = [];
    mixture_covariance_mat = mixture_variances_mat;
    
    
else
    %Multivariate case.
    
    %get constants, set up vectors:
    data = Z_input_data_mat;
    [number_of_samples,number_of_dimensions] = size(data);
    max_data = max(data);	%gives max in each dimension
    min_data = min(data);
    mixture_means_mat = zeros(number_of_dimensions,max_number_of_mixture_terms);	%each col is a term
    mixture_variances_mat = zeros(number_of_dimensions,max_number_of_mixture_terms*number_of_dimensions);	%each dxd submatrix is term
    mixture_pies_vec = zeros(1,max_number_of_mixture_terms);
    posterior = zeros(1,max_number_of_mixture_terms);
    min_bound_on_new_pie = 0.00001;		% lower bound on new pies
    maximum_bound_on_variance = 1000;		% bounding the parameter space
    
    %Initialize density to first data point:        
    number_of_mixture_terms = 1;
    mixture_means_mat(:,1) = data(1,:)';
    mixture_variances_mat(:,1:number_of_dimensions) = eye(number_of_dimensions,number_of_dimensions);
    mixture_pies_vec(1) = 1; 
    for sample_counter = 2:number_of_samples
        flag_create_new_term = statistics_adaptive_mixtures_return_whether_to_create_new_term(data(sample_counter),mixture_pies_vec,mixture_means_mat,mixture_variances_mat,new_cluster_creation_threshold,number_of_mixture_terms,max_number_of_mixture_terms);
        if ~flag_create_new_term	
            %Update terms:
            posterior(1:number_of_mixture_terms) = statistics_adaptive_mixtures_get_posterior_probabilities2(data(sample_counter,:)',mixture_pies_vec,mixture_means_mat,mixture_variances_mat,number_of_mixture_terms);
            mixture_variances_mat = rvarupm(data(sample_counter,:),mixture_pies_vec,mixture_means_mat,mixture_variances_mat,posterior,number_of_mixture_terms,sample_counter);
            mixture_means_mat = rmuupm(data(sample_counter,:),mixture_pies_vec,mixture_means_mat,posterior,number_of_mixture_terms,sample_counter);
            mixture_pies_vec(1:number_of_mixture_terms) = rpieupm(posterior,mixture_pies_vec,sample_counter,number_of_mixture_terms);
            
        else
            %Create a new term:
            number_of_mixture_terms = number_of_mixture_terms+1;
            mixture_means_mat(:,number_of_mixture_terms) = data(sample_counter,:)';
            mixture_pies_vec(number_of_mixture_terms) = max([1/(sample_counter),min_bound_on_new_pie]);	% update pies
            mixture_pies_vec(1:number_of_mixture_terms-1) = mixture_pies_vec(1:number_of_mixture_terms-1)*(1-mixture_pies_vec(number_of_mixture_terms));
            mixture_variances_mat(:,number_of_dimensions*(number_of_mixture_terms-1)+1:number_of_dimensions*number_of_mixture_terms) = ....
                   setvarm(mixture_means_mat,mixture_pies_vec,mixture_variances_mat,data(sample_counter,:),number_of_mixture_terms-1);
        end 	% end if create statement            
    end  % for i loop
    %Clean up the arrays:
    mixture_means_mat(:,(number_of_mixture_terms+1):max_number_of_mixture_terms) = [];
    mixture_pies_vec((number_of_mixture_terms+1):max_number_of_mixture_terms) = [];
    %Note that each page is a covariance matrix for a term:
    mixture_covariance_mat = zeros(number_of_dimensions,number_of_dimensions,number_of_mixture_terms);
    for term_counter = 1:number_of_mixture_terms
        mixture_covariance_mat(:,:,term_counter) = mixture_variances_mat(:,((term_counter-1)*number_of_dimensions+1):term_counter*number_of_dimensions);
    end
end

    