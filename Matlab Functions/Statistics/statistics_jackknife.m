function [bhat,sehat,jvals] = statistics_jackknife(data,fname)
% CSJACK	Jackknife estimate of bias and standard error.
%
%	[BHAT, SEHAT, JVALS] = CSJACK(X,FNAME)
%
%	This function returns the jackknife estimate of bias and standard
%	error for a statistic calculated by the function FNAME. 
%
%	X is an n x d array, where each row is an observation.
%	FNAME is a string for a MATLAB function that calculates a
%	statistic. The input argument to FNAME must be X only.
%	BHAT is the jackknife estimate of bias.
%	SEHAT is the jackknife estimate of the standard error.
%	JVALS is a vector of jackknife replicates.
%
%	See also CSBOOT, CSJACKboot
%
%	Example: 	x = randn(50,1);
%				[bhat,sehat,jvals] = csjack(x,'mean');
%				[bhat,sehat,jvals] = csjack(x,'myfun');


%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox


[number_of_samples,number_of_dimensions] = size(data);
if number_of_dimensions > number_of_samples
	error('Each row of the data array must correspond to an observation.')
end

%Find the observed value of the statistic:
thetahat = feval(fname,data);
jvals = zeros(number_of_samples,1);
for i = 1:number_of_samples
   % use feval to evaluate the estimate with the i-th obervation removed
   % These are the jackknife replications.
   jvals(i) = feval(fname, [data(1:(i-1),:);data((i+1):number_of_samples,:)]);
end
bhat = (number_of_samples-1)*(mean(jvals)-thetahat); 
sehat = sqrt((number_of_samples-1)/number_of_samples*sum((jvals-mean(jvals)).^2));




