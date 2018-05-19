function y = statistics_get_poisson_cumulative_distribution_at_points(x,lambda_parameter)
% CSPOISC Poisson cumulative distribution function.
%
%   Y = CSPOISC(X,LAMBAD) Returns the value of the
%   Poisson cumulative distribution function with
%   parameter LAMBDA at the values given in X.
%
%   See also CSPOISP, CSPOIRND, CSPOIPAR, CSPOISSPLOT


%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox 


if lambda_parameter <= 0
   error('Lambda must be greater than 0')
   return
end

% CHECK FOR DISCRETE X VALUES
if any(find(floor(x) ~= x))
   error('X values must be discrete')
   return
end

y = zeros(size(x));  

% find the values, sum up the previous probabilities
for i = 1:length(x)
   if x(i) >=0 
      y(i)=sum(statistics_get_poisson_probability_function_at_points(0:x(i),lambda_parameter));
   end
end
