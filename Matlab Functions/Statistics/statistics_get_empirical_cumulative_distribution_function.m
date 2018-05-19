function fx = statistics_get_empirical_cumulative_distribution_function(input_data_vec,x)
% CSECDF Empirical cumulative distribution function.
%
%   FHAT = CSECDF(DATA,X) Returns the empirical
%   distribution function at a given vector of X
%   locations, using the sample contained in DATA.

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox


number_of_input_samples = length(input_data_vec);
input_data_vec_sorted = sort(input_data_vec);
fx = zeros(size(x));

for i = 1:length(x)
    ind = find(input_data_vec_sorted<=x(i));
    if ~isempty(ind)
        fx(i) = length(ind)/number_of_input_samples;
    else
        fx(i) = 0;
    end
end
