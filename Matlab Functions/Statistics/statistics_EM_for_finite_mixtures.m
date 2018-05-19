function [mixtures_pies_vec_current,mixtures_means_mat_current,mixtures_variances_mat_current] = statistics_EM_for_finite_mixtures(...
                                        Z_input_data_mat, ...
                                        mixtures_means_mat_previous, ...
                                        mixtures_variances_mat_previous, ...
                                        mixtures_pies_vec_previous, ...
                                        max_number_of_iterations, ...
                                        convergence_tolerance)
% CSFINMIX  Expectation-Maximization algorithm for finite mixtures.
%
%   [WTS,MUS,VARS] = CSFINMIX(DATA,MUIN,VARIN,WTSIN,MAXIT,TOL)
%
%   This function implements the EM algorithm for estimating finite mixtures.
%   It requires an initial model for the component densities. This assumes
%   component densities are normals (univariate or multivariate). 
%
%   INPUTS: DATA is a matrix of observations, one on each row.
%           MUIN is an array of means, each column corresponding to a mean.
%           VARIN is a vector of variances in the univariate case. In the
%           multivariate case, it is a 3-D array of covariance matrix, one
%           page per component density.
%           WTSIN is a vector of weights.
%           MAXIT is the maximum allowed number of iterations.
%           TOL is the convergence tolerance.
%
%   An example of a bivariate. 2-term VARIN is:
%           varin(:,:,1) = 2*eye(2);
%           varin(:,:,2) = 3*eye(2);

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox

[number_of_samples,number_of_dimensions] =  size(Z_input_data_mat);
if number_of_samples==1 || number_of_dimensions==1
    %then it is univariate data:
    Z_input_data_mat = Z_input_data_mat(:);
    number_of_dimensions = 1;
end
number_of_term = length(mixtures_pies_vec_previous);

if number_of_dimensions == 1
    %Initialize the vectors/matrices:
    [number_of_samples,number_of_dimensions] = size(Z_input_data_mat);
    %posterior(i,j) = probability the ith data point was drawn from jth term
    posterior_probabilities_per_sample_mat = zeros(number_of_samples,number_of_term);  
    iteration_counter = 1;
    tolerance_current = convergence_tolerance + 1; %to be able to enter loop
    
    %Loop over different iterations:
    while iteration_counter <= max_number_of_iterations && tolerance_current > convergence_tolerance;
        %Update parameters:
        posterior_probabilities_per_sample_mat = update_posterior_probabilities(Z_input_data_mat,mixtures_means_mat_previous,mixtures_variances_mat_previous,mixtures_pies_vec_previous);
        mixtures_pies_vec_current = update_mixture_pies(posterior_probabilities_per_sample_mat);
        mixtures_means_mat_current = update_mixture_means(Z_input_data_mat,posterior_probabilities_per_sample_mat,mixtures_pies_vec_current);
        mixtures_variances_mat_current = update_mixture_variances(Z_input_data_mat,posterior_probabilities_per_sample_mat,mixtures_pies_vec_current,mixtures_means_mat_current);
        
        %Check results and update:
        tolerance_current = max([abs(mixtures_pies_vec_previous-mixtures_pies_vec_current), ...
                      abs(mixtures_means_mat_current-mixtures_means_mat_previous),...
                      abs(mixtures_variances_mat_current-mixtures_variances_mat_previous)]);
        iteration_counter = iteration_counter+1;
        mixtures_pies_vec_previous = mixtures_pies_vec_current;
        mixtures_means_mat_previous = mixtures_means_mat_current;
        mixtures_variances_mat_previous = mixtures_variances_mat_current;
    end    % while loop
    
else
    %Multi-Variate:
    
    %reset the parameters to the right names:
    mixtures_means_mat_previous = mixtures_means_mat_previous;
    mixtures_variances_mat_previous = mixtures_variances_mat_previous;
    mixtures_pies_vec_previous = mixtures_pies_vec_previous;
    
    %get necessary parameters:
    convergence_tolerance = 0.001;
    [number_of_samples,number_of_dimensions] = size(Z_input_data_mat);	% n=# pts, d=# dims
    number_of_mixture_terms = length(mixtures_pies_vec_previous);	% c=# terms
    
    %Initialize the vectors/matrices:
    % posterior(i,j)=probability the ith obsv is drawn from jth term
    iteration_counter = 1;
    tolerance_current = convergence_tolerance+1;	% to get started
    while iteration_counter <= max_number_of_iterations && tolerance_current > convergence_tolerance
                
        posterior_probabilities_per_sample_mat = update_posterior_probabilities_multivariate(Z_input_data_mat,mixtures_means_mat_previous,mixtures_variances_mat_previous,mixtures_pies_vec_previous);
        mixtures_pies_vec_current = update_mixture_pies_multivariate(posterior_probabilities_per_sample_mat);
        mixtures_means_mat_current = update_mixture_means_multivariate(Z_input_data_mat,posterior_probabilities_per_sample_mat,mixtures_pies_vec_previous);
        mixtures_variances_mat_current = update_mixture_variance_multivariate(Z_input_data_mat,posterior_probabilities_per_sample_mat,mixtures_pies_vec_previous,mixtures_means_mat_previous);
        delta_variance_per_mixture_term_vec = zeros(1,number_of_mixture_terms);
        for j = 1:number_of_mixture_terms
            delta_variance_per_mixture_term_vec(j) = ...
                                max(max(abs(mixtures_variances_mat_current(:,:,j)-mixtures_variances_mat_previous(:,:,j))));
        end
        delmu = max(max(abs(mixtures_means_mat_current-mixtures_means_mat_previous)));
        delpi = max(mixtures_pies_vec_previous-mixtures_pies_vec_current);
        tolerance_current = max([max(delta_variance_per_mixture_term_vec),delmu,delpi]);
        
        %reset parameters:
        iteration_counter = iteration_counter+1;
        mixtures_pies_vec_previous = mixtures_pies_vec_current;
        mixtures_means_mat_previous = mixtures_means_mat_current;
        mixtures_variances_mat_previous = mixtures_variances_mat_current;
    end  % while loop
    
    
end


% function posterior=postup(data,mu,var,mix_cof)
function posterior = update_posterior_probabilities(Z_input_data_mat,mixtures_means_mat,mixtures_variances_mat,mixtures_pies_vec)

number_of_terms = length(mixtures_means_mat);	% number of terms
total_probability = zeros(size(Z_input_data_mat));
for i = 1:number_of_terms	%loop to find total prob in denominator (hand, pg 37)
  posterior(:,i) = (mixtures_pies_vec(i)/sqrt(mixtures_variances_mat(i))) * ...
                            exp(-(Z_input_data_mat-mixtures_means_mat(i)).^2/(2*mixtures_variances_mat(i)));
  total_probability = total_probability + posterior(:,i);
end
posterior_probabilities_sum = total_probability*ones(1,number_of_terms);  % this should work!!!
posterior = posterior./posterior_probabilities_sum;

% piup.m
%function mix_cof=piup(posterior)
function mixture_pies = update_mixture_pies(posterior_probabilities_per_sample_mat)
[number_of_samples,number_of_clusters] = size(posterior_probabilities_per_sample_mat);
mixture_pies = sum(posterior_probabilities_per_sample_mat)/number_of_samples;

% function mu=muup(data,posterior,mix_cof)
function mixture_means_mat = update_mixture_means(Z_input_data_mat,posterior_probabilities_per_sample_mat,mixture_pies_vec)
[number_of_samples,number_of_clusters] = size(posterior_probabilities_per_sample_mat);
mixture_means_mat = Z_input_data_mat'*posterior_probabilities_per_sample_mat;
mixture_means_mat = mixture_means_mat./mixture_pies_vec;
mixture_means_mat = mixture_means_mat/number_of_samples;

% function var=varup(data,posterior,mix_cof,mu)
function mixture_variances_vec = update_mixture_variances(Z_input_data_mat,posterior_probabilities_per_sample_mat,mixture_pies_vec,mixture_means_mat)
[number_of_samples,number_of_mixtures] = size(posterior_probabilities_per_sample_mat);
Z_input_data_mat = Z_input_data_mat*ones(1,number_of_mixtures) - ones(number_of_samples,1)*mixture_means_mat;
Z_input_data_mat = Z_input_data_mat.^2;
mixture_variances_vec = (diag(posterior_probabilities_per_sample_mat'*Z_input_data_mat))';
mixture_variances_vec = mixture_variances_vec./mixture_pies_vec;
mixture_variances_vec = mixture_variances_vec/number_of_samples;

% function posterior=postupm(data,mu,var_mat,mix_cof)
function posterior = update_posterior_probabilities_multivariate(Z_input_data_mat,mixture_means_mat,mixture_variances_mat,mixture_pies_vec)

[number_of_samples,number_of_dimensions] = size(Z_input_data_mat);
[number_of_dimensions,number_of_clusters] = size(mixture_means_mat);
total_probability = zeros(number_of_samples,1);	% need one per data point, denon of eq 2.19, pg 37
for i = 1:number_of_clusters	%loop to find total prob in denominator (hand, pg 37)
  posterior(:,i) = mixture_pies_vec(i) * ...
                        statistics_get_multivariate_normal_density_at_points(...
                                Z_input_data_mat,...
                                mixture_means_mat(:,i)',...
                                mixture_variances_mat(:,(i-1)*number_of_dimensions+1 : i*number_of_dimensions));
  total_probability = total_probability + posterior(:,i);
end

posterior_probabilities_sum = total_probability*ones(1,number_of_clusters);  % this should work!!!
posterior = posterior./posterior_probabilities_sum;

% function var_mat=varupm(data,posterior,mix_cof,mu)
function mixture_variances_mat = update_mixture_variance_multivariate(Z_input_data_mat,posterior_probabilities_per_sample_mat,mixtures_pies_vec,mixtures_means_mat)

[number_of_samples,number_of_clusters] = size(posterior_probabilities_per_sample_mat);
[number_of_samples,number_of_dimensions] = size(Z_input_data_mat);
mixture_variances_mat = zeros(number_of_dimensions,number_of_dimensions,number_of_clusters);
for i = 1:number_of_clusters
    centered_data = Z_input_data_mat - ones(number_of_samples,1)*mixtures_means_mat(:,i)';
    cluster_covariance_mat = centered_data'*diag(posterior_probabilities_per_sample_mat(:,i))*centered_data;
    mixture_variances_mat(:,:,i) = cluster_covariance_mat./(mixtures_pies_vec(i)*number_of_samples);
end

%function mix_cof=piupm(posterior)
function mixture_pies_vec = update_mixture_pies_multivariate(posterior)
[number_of_samples,number_of_clusters] = size(posterior);
mixture_pies_vec = sum(posterior)/number_of_samples;

% function mu=muupm(data,posterior,mix_cof)
function mixture_means_mat = update_mixture_means_multivariate(Z_input_data_mat,posterior_probabilities_per_sample_mat,mixture_pies_vec)
[number_of_sample,number_of_clusters] = size(posterior_probabilities_per_sample_mat);
[number_of_samples,number_of_dimensions] = size(Z_input_data_mat);
mixture_means_mat = Z_input_data_mat'*posterior_probabilities_per_sample_mat;
MIX = ones(number_of_dimensions,1)*mixture_pies_vec;
mixture_means_mat = mixture_means_mat./MIX;
mixture_means_mat = mixture_means_mat/number_of_sample;


