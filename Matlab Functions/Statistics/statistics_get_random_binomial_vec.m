function X = statistics_get_random_binomial_vec(number_of_trials_for_the_variables,p_probability,number_of_variables)
% CSBINRND Returns a vector of binomial random variates.
%
%   X = CSBINRND(TRIALS,P,N) returns a vector of N random
%   variables from the binomial distribution with parameters
%   TRIALS and success probability P.
%
%   See also CSBINOP, CSBINOC, CSBINPAR, CSBINOQ   


%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox 


% Generate the uniform random numbers
% N variates of n trials
U = rand(number_of_trials_for_the_variables,number_of_variables);
% Add up all of those less than or equal to
% the success probability.
X = sum(U<=p_probability);


