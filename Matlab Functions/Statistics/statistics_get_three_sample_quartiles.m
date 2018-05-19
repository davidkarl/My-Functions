function q = statistics_get_three_sample_quartiles(input_vec)

%   QUARTILES   Finds the three sample quartiles
%
%   Q = quartiles(X)
%   This returns the three sample quartiles as defined by Tukey,
%   Exploratory Data Analysis, 1977.

% First sort the data.
input_vec = sort(input_vec);
% Get the median.
q2 = median(input_vec);
% First find out if n is even or odd.
n = length(input_vec);
if rem(n,2) == 1
    odd = 1;
else
    odd = 0;
end
if odd
    q1 = median(input_vec(1:(n+1)/2));
    q3 = median(input_vec((n+1)/2:end));
else
    q1 = median(input_vec(1:n/2));
    q3 = median(input_vec(n/2:end));
end
q(1) = q1;
q(2) = q2;
q(3) = q3;