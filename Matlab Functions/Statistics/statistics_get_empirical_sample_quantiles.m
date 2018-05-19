function qhat = statistics_get_empirical_sample_quantiles(x_given_probability_density_samples,probability_densities)
% CSQUANTILES Sample quantiles.
%
%	QHAT = CSQUANTILES(X,P) Returns the sample quantiles
%	for the probabilities given in P using the sample in
%	X.

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox 


if ~isempty(find(probability_densities <= 0 | probability_densities >1))
   error('Probabilities must be between 0 and 1')
   return
end
xs = sort(x_given_probability_density_samples);
qhat = zeros(size(probability_densities));
n=length(x_given_probability_density_samples);
phat = ((1:n)-0.5)/n;
%for i=1:length(p)
 %  if p(i)~=0
 %     ind=find(p(i)<=phat);
 %     qhat(i)=x(ind(1));
 %  elseif  p(i)==0
 %     qhat(i)=nan;
 %  end
 %end
 % use linear interpolation to find the quantiles.
 % think of the phat as the x's and the order statistics as the
 % y's. We will use the definition that the j-th order statistic
 % is an estimate of the (j-0.5)/n quantile. If we want some other
 % quantile, then we can use linear interpolation.
qhat=interp1(phat,xs,probability_densities);
      