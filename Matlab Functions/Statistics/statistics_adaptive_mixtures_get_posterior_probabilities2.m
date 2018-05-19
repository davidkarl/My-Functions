function posterior = statistics_adaptive_mixtures_get_posterior_probabilities2(x,pies,mus,vars,nterms)
% RPOSTUPM  Helper function for adaptive mixtures.

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox


totprob=0;
posterior=zeros(1,nterms);

[d,c]=size(mus);

for i=1:nterms	%loop to find total prob in denominator (hand, pg 37)
  posterior(i)=pies(i)*statistics_get_multivariate_normal_density_at_points(x',mus(:,i)',vars(:,(i-1)*d+1:i*d));
  totprob=totprob+posterior(i);
end


posterior=posterior/totprob;



