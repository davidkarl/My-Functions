function [clabs, err] = statistics_classify_data_from_mixture_model(Z_input_data,...
                                                                    mixture_pies_vec,...
                                                                    mixture_means_mat,...
                                                                    mixture_variances_mat)

% MIXCLASS  Get the classification from a mixture model.
%
%   [CLABS,ERR] = MIXCLASS(DATA,WGTS,MUS,VARS)
%
%   For a given set of DATA (nxd) and a mixture model given by
%   WGTS (weights), MUS (component means), and VARS (component
%   variances), return the class labels in CLABS, along with
%   the associated classification error in ERR.
%

%   Model-based Clustering Toolbox, January 2003


% set up the space for the output vectors.
[n,d] = size(Z_input_data);
err = zeros(1,n);
clabs = zeros(1,n);
for i = 1:n     
    sample_posteriori_probabilities = get_posterior_probability_multivariate(Z_input_data(i,:)',mixture_pies_vec,mixture_means_mat,mixture_variances_mat);
    [v, clabs(i)] = max(sample_posteriori_probabilities);     % classify it with the highest posterior prob.
    err(i) = 1 - v;     % Classification error is 1 - posterior
end

%%%%%%%%%%%%FUNCTION - POSTM %%%%%%%%%%%%%%%%%%%%%%
function sample_posterior_probabilities = get_posterior_probability_multivariate(x_sample,mixture_pies_vec,mixture_means_mat,mixture_variances_mat)
number_of_mixture_terms = length(mixture_pies_vec);
posterior_probability_sum = 0;
sample_posterior_probabilities = zeros(1,number_of_mixture_terms);
for i = 1:number_of_mixture_terms	%loop to find total prob in denominator (hand, pg 37)
    sample_posterior_probabilities(i) = mixture_pies_vec(i)*evalnorm(x_sample',mixture_means_mat(:,i)',mixture_variances_mat(:,:,i));
    posterior_probability_sum = posterior_probability_sum+sample_posterior_probabilities(i);
end
sample_posterior_probabilities = sample_posterior_probabilities/posterior_probability_sum;


%%%%%%%%%%%%%%%  FUNCTION EVALNORM %%%%%%%%%%%%%
function prob = evalnorm(x,mu,cov_mat)
[n,d]=size(x);
prob = zeros(n,1);
a=(2*pi)^(d/2)*sqrt(det(cov_mat));
covi = inv(cov_mat);
for i = 1:n
	xc = x(i,:)-mu;
	arg=xc*covi*xc';
	prob(i)=exp((-.5)*arg);
end
prob=prob/a;
