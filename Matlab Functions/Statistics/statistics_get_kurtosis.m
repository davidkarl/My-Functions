function beta4 = statistics_get_kurtosis(input_vec)
% CSKURTOSIS Coefficient of kurtosis.
%
%   GAM = CSKURTOSIS(X) Returns the sample coefficient of
%   kurtosis using the sample in X. The kurtosis should
%   be approximately 3 for a normal distribution.
%
%   See also CSKEWNESS

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox 



number_of_samples = length(input_vec);
mu = mean(input_vec);

num = (1/number_of_samples)*sum((input_vec-mu).^4);
den = (1/number_of_samples)*sum((input_vec-mu).^2);

beta4 = num/den^2;