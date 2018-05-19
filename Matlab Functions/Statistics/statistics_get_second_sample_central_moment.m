function mr = statistics_get_second_sample_central_moment(input_vec)
% MOM   Sample second central moment.
%
%   MR = MOM(X)
%   This function returns the sample second central moment.


%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox

n = length(input_vec);
mu = mean(input_vec);
mr = (1/n)*sum((input_vec-mu).^2);
