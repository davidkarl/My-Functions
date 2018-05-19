function [blo,bhi,bvals] = statistics_bootstrap_percentile_interval(data,fname,number_of_bootstrap_replicates,alpha)
% CSBOOPERINT Boostrap percentile interval.
%
%	[BLO,BHI,BVALS] = CSBOOPERINT(X,FNAME,B,ALPHA)
%
%	This function determines the bootstrap percentile interval
%	for a statistic given by FNAME. The output from this function
%	is the lower and upper endpoints of a (1-ALPHA)*100% confidence
%	interval using the bootstrap percentile method.
%
%	X is an n x d array, where each row is an observation.
%	FNAME is a string for a MATLAB function that calculates a
%	statistic. The input argument to FNAME must be X only.
%	B is the number of bootstrap replicates. It is recommended that
%	this number be greater than 1000 for confidence intervals.
%	BLO and BHI are the endpoints of the interval.
%	BVALS is a vector of bootstrap replicates. 
%
%	See also CSBOOT, CSBOOTINT, CSBOOTBCA
%
%	Example: 	x = randn(50,1);
%				[blo,bhi,bvals] = csbooperint(x,'mean',1000,0.05);


%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox

if alpha >= 0.5
	error('Alpha must be less than 0.5')
end

[n,d] = size(data);
if d > n
	error('Each row of the data array must correspond to an observation.')
end

bvals = zeros(number_of_bootstrap_replicates,1);
% Loop over each resample and calculate the bootstrap replicates
for i = 1:number_of_bootstrap_replicates
   % generate the indices for the B bootstrap resamples, sampling with
   % replacement using the discrete uniform distribution.
   ind = ceil(n.*rand(n,1));
   % extract the sample from the data 
   % each row corresponds to a bootstrap resample
   xstar = data(ind,:);
   % use feval to evaluate the estimate for the i-th resample
   bvals(i) = feval(fname, xstar);
end
% Assumes alpha < 0.5 and uses convention in Efron & Tibshirani, 1993, pg. 160
k = floor(((number_of_bootstrap_replicates+1)*alpha/2));
sbval = sort(bvals);
blo = sbval(k);
bhi = sbval(number_of_bootstrap_replicates+1-k);
