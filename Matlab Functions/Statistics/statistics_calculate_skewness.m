function beta3 = statistics_calculate_skewness(input_vec)
% CSKEWNESS Coefficient of skewness.
%
%   GAM = CSKEWNESS(X) Returns the sample coefficient
%   of skewness using the sample in X.
%
%   See also CSKURTOSIS

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox 



number_of_samples = length(input_vec);
mu = mean(input_vec);
num = (1/number_of_samples)*sum((input_vec-mu).^3);
den = (1/number_of_samples)*sum((input_vec-mu).^2);

beta3 = num/den^(3/2);