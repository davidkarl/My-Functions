function y_probability = statistics_get_poisson_probability_function_at_points(x_values,lambda_parameter)
% CSPOISP Poisson probability mass function.
%
%   Y = CSPOISP(X,LAMBDA) Returns the value of the
%   probability mass function for the Poisson 
%   distribution with parameter LAMBDA at the 
%   values given in X.
%
%   See also CSPOISC, CSPOIRND, CSPOIPAR, CSPOISSPLOT


%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox 


% CHECK FOR DISCRETE X VALUES
if any(find(floor(x_values) ~= x_values))
   error('X values must be discrete')
   return
end

if lambda_parameter <= 0
   error('Lambda must be greater than 0')
   return
end
y_probability=zeros(size(x_values));
ind=find(x_values>=0);
y_probability(ind) = exp(-lambda_parameter)*lambda_parameter.^(x_values(ind))./gamma(x_values(ind)+1);