function X = statistics_get_random_multivariate_normal_samples(mu_vec,cov_mat,number_of_samples)
% CSMVRND Generate multivariate normal random variables.
%
%   R = CSMVRND(MU,COVM,N) Generates a sample of size N
%   random variables from the multivariate normal 
%   distribution. MU is the d-dimensional mean as a 
%   column vector. COVM is the d x d covariance matrix.
%

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox 


if det(cov_mat) <=0
    % Then it is not a valid covariance matrix.
    error('The covariance matrix must be positive definite')
end

mu_vec = mu_vec(:); % Just in case it is not a column vector.
number_of_dimensions = length(mu_vec);

%get cholesky factorization of covariance:
R = chol(cov_mat);

%generate the standard normal random variables:
Z = randn(number_of_samples,number_of_dimensions);
X = Z*R + ones(number_of_samples,1)*mu_vec';