function piess = statistics_adaptive_mixtures_update_pies2(posterior,pies,n,nterms)
% RPIEUPM   Helper function for adaptive mixtures.

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox


betan=1/(n);
post=posterior(1:nterms);
piess=pies(1:nterms);
piess=piess+betan*(post-piess);


