function x = statistics_get_binomial_distribution_quantiles(quantile_probability_values_vec,number_of_trials,p_binumial_probability)
% CSBINOQ Quantiles of the binomial distribution.
%
%   X = CSBINOQ(PROB,TRIALS,P) calculates the inverse
%   (or quantiles) of the binomial cumulative distribution 
%   function with parameters TRIALS and success probability P
%   at a given vector of probability values PROB.
%
%   See also CSBINOP, CSBINOC, CSBINRND, CSBINPAR    


%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox 

% quantile_probability_values_vec = 0.5;
% number_of_trials = 10;
% p_binumial_probability = 0.5;

if p_binumial_probability < 0 || p_binumial_probability > 1
   error('Success probability must be between 0 and 1')
   return
end

if ~isempty(find(quantile_probability_values_vec<0 | quantile_probability_values_vec >1))
   error('Probabilities must be between 0 and 1')
   return
end

x = zeros(size(quantile_probability_values_vec));

%Find all values of the cdf over 0:n:
t = 0:number_of_trials;
cumulative_distribution_probability = ...
                statistics_get_binomial_cumulative_distribution_probability(t,number_of_trials,p_binumial_probability);
for i = 1:length(quantile_probability_values_vec)
   ind = find(cumulative_distribution_probability <= quantile_probability_values_vec(i));
   if ~isempty(ind) 
       x(i) = t(length(ind));
   end
end







