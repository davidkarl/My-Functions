function y = statistics_get_uniform_cumulative_distribution_probability(input_vec,a,b)
% CSUNIFC Uniform cumulative distribution function.
%
%	Y = CSUNIFC(X,A,B) Returns the values of the uniform
% 	cumulative distribution function with parameters
%	A and B at the domain locations given in X.
%
%	See also CSUNIFP, CSUNIFQ, CSUNIPAR

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox


if a > b
   error('A must be less than B.')
   return
end

y = zeros(size(input_vec)); 
ind = find(input_vec>=b);
y(ind)=1;
ind = find(input_vec>=a & input_vec<=b);
y(ind)=(input_vec(ind)-a)/(b-a);
