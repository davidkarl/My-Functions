function y = statistics_get_binomial_probability_for_number_of_successes(number_of_successes_vec,number_of_trials,p_binomial_probability)
% CSBINOP Binomial probability mass function.
%
%   Y = CSBINOP(X,TRIALS,P) Returns the value of the probability
%   mass function for the binomial distribution with parameters
%   TRIALS and success probability P, at the values given in the
%   vector X.
%
%   See also CSBINOC, CSBINRND, CSBINPAR, CSBINOQ   

%   W. L. and A. R. Martinez, 4/10/00
%   Computational Statistics Toolbox

p_binomial_probability = 0.3;
number_of_trials = 10;
number_of_successes_vec = number_of_trials/2;

if p_binomial_probability < 0 || p_binomial_probability > 1
   error('Success probability must be between 0 and 1')
   return
end

% CHECK FOR DISCRETE X VALUES
if any(find(floor(number_of_successes_vec) ~= number_of_successes_vec))
   error('X values must be discrete')
   return
end

% find the values
y = zeros(size(number_of_successes_vec));
ind = find(number_of_successes_vec>=0 & number_of_successes_vec<=number_of_trials); %defined only for this range
for i = ind
   y(i) = number_of_combinations_x_out_of_n(number_of_trials,number_of_successes_vec(i));
end
y = y.*(p_binomial_probability.^number_of_successes_vec).*(1-p_binomial_probability).^(number_of_trials-number_of_successes_vec);
   

function com = number_of_combinations_x_out_of_n(n,x)
% this function finds the combination of n things taken x at a time

df = n-x;
tmp = max([df,x])+1;
num = prod(n:-1:tmp) ;
den = prod(2:min([df, x]));
com = num/den;






