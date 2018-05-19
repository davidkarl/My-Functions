function y = statistics_get_binomial_cumulative_distribution_probability(number_of_successes_vec,number_of_trials,p_binomial_probability)
% CSBINOC Binomial cumulative distribution function.
%
%   Y = CSBINOC(X,TRIALS,P) returns the value of the binomial 
%   cumulative distribution function with parameters
%   TRIALS and success probability P at
%   given vector of X values.
%
%   See also CSBINOP, CSBINRND, CSBINPAR, CSBINOQ   


%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox 


if p_binomial_probability < 0 || p_binomial_probability > 1
   error('Success probability must be between 0 and 1')
   return
end

% CHECK FOR DISCRETE X VALUES
if any(find(floor(number_of_successes_vec) ~= number_of_successes_vec))
   error('X values must be discrete');
   return
end

y = zeros(size(number_of_successes_vec));  % NOTE: x < 0 yields y = 0
ind = find(number_of_successes_vec > number_of_trials);
if ~isempty(ind)  % these get a value of 1
   y(ind) = 1;
end

ind = find(number_of_successes_vec>=0 & number_of_successes_vec<=number_of_trials);
for i = ind
      y(i) = sum(statistics_get_binomial_probability_for_number_of_successes(...
                                                 0:number_of_successes_vec(i),number_of_trials,p_binomial_probability) );
end
