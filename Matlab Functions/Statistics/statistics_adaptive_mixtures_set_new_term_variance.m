function newvar = statistics_adaptive_mixtures_set_new_term_variance(means_mat,pies_vec,variances_mat,Z_input_data,number_of_mixture_terms)
% CSSETVAR  Set new variance in adaptive mixtures density estimation.
%
%   NEWVAR = CSSETVAR(MUS,WGTS,VARS,X,NTERMS)
%   This function will set the variance for a new term in univariate
%   adaptive mixtures density estimation. NTERMS must be
%   the current number of terms and does not include the new term.

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox



f=exp(-.5*(Z_input_data-means_mat(1:number_of_mixture_terms)).^2./variances_mat(1:number_of_mixture_terms)).*pies_vec(1:number_of_mixture_terms);
f=f/sum(f);
f=f.*variances_mat(1:number_of_mixture_terms);
newvar=sum(f);

