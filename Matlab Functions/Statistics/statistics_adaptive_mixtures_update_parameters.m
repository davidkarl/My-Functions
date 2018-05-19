function [piess,muss,varss] = statistics_adaptive_mixtures_update_parameters(...
                                                           Z_input_mat, ...
                                                           pies_vec, ...
                                                           means_mat, ...
                                                           variances_mat, ...
                                                           posterior_probabilities, ...
                                                           number_of_mixture_terms, ...
                                                           iteration_and_sample_counter)
% CSRUP     Recursive updates of finite mixture parameters.
%
%   [WGTS,MUS,VARS] = CSRUP(X,WGTS,MUS,VARS,POST,NTERMS,N)
%
%   This function recursively updates all of the parameters for
%   the adaptive mixtures density estimation approach. These are
%   the recursive update equations for the EM algorithm. 

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox


inertvar=10;
betan=1/(iteration_and_sample_counter);
piess=pies_vec(1:number_of_mixture_terms);
muss=means_mat(1:number_of_mixture_terms);
varss=variances_mat(1:number_of_mixture_terms);
post=posterior_probabilities(1:number_of_mixture_terms);
% update the mixing coefficients
piess=piess+(post-piess)*betan;
% update the means
muss=muss+betan*post.*(Z_input_mat-muss)./piess;
% update the variances
denom=(1/betan)*piess+inertvar;
varss=varss+post.*((Z_input_mat-muss).^2-varss)./denom;


