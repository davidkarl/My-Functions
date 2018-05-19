function [a,b] = statistics_estimate_uniform_distribution_parameters(input_vec)
% CSUNIPAR Estimate the parameters of the uniform distribution.
%
%	[A,B] = CSUNIPAR(X) Returns the an estimate of the parameters 
%	A and B for the uniform disribution. The random sample is
%	contained in X.
%
%	See also CSUNIFP, CSUNIFC, CSUNIFQ

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox

a=min(input_vec);
b=max(input_vec);

