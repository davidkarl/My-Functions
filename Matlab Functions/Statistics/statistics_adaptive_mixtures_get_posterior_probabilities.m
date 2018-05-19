function post = statistics_adaptive_mixtures_get_posterior_probabilities(Z_data_input,pies_vec,means_mat,variances_mat,number_of_mixutre_terms)
% CSRPOSTUP Posterior probabilities.
%
%   POST = CSRPOSTUP(X,WGTS,MUS,VARS,NTERMS)
%   This function will return the posterior probabilities
%   for a univariate finite mixture.

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox

f=exp(-0.5*(Z_data_input-means_mat(1:number_of_mixutre_terms)).^2./variances_mat(1:number_of_mixutre_terms)) ...
            .* pies_vec(1:number_of_mixutre_terms);
f=f/sum(f);
post=f;

