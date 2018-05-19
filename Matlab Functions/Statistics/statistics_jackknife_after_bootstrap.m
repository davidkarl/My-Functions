function [bhat,sehatT,varhatB,bvals] = statistics_jackknife_after_bootstrap(data,fname,B)
% CSJACKBOOT	Jackknife-after-bootstrap
%
%	[BHAT,SEHATT,SEHATB,BVALS] = CSJACKBOOT(DATA,FNAME,B)
%
%	This function will return the bootstrap distribtuion for
%	a statistic given by FNAME.
%
%	X is an n x d array, where each row is an observation.
%	FNAME is a string for a MATLAB function that calculates a
%	statistic. The input argument to FNAME must be X only.
%	B is the number of bootstrap replicates. It is recommended that
%	this number be greater than 1000.
%	BHAT is the bootstrap estimate of bias of the statistic FNAME.
%	SEHATT is the bootstrap estimate of the variance of the statistic FNAME.
%	VARHATB is the bootstrap estimate of the standard error in the estimate of SEHATT.
%	BVALS is the vector of bootstrap replicates.
%
%	See also CSBOOT, CSJACK
%
%	Example: 	x = randn(50,1);
%				[bhat,sehatT,varhatB,bvals] = csbootjack(x,'mean',1000);
%				[bhat,sehatT,varhatB,bvals] = csbootjack(x,'myfun',1000);


%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox

% Check to make sure that each row of the data matrix is 
% an observation.
[n,d] = size(data);
if d > n
	error('Each row of the data array must correspond to an observation.')
end

% FIRST GET THE BOOTSTRAP INFORMATION.
% find observed value of statistic using original data
thetahat = feval(fname,data);
bvals = zeros(B,1);
bootsam = zeros(n, B);	% Each col has the indices for the bootstrap sample.
% Loop over each resample and calculate the bootstrap replicates
for i = 1:B
   % generate the indices for the B bootstrap resamples, sampling with
   % replacement using the discrete uniform distribution.
   ind = ceil(n.*rand(n,1));
   % Save the indices.
   bootsam(:,i) = ind(:);
   % extract the sample from the data 
   % each row corresponds to a bootstrap resample
   xstar = data(ind,:);
   % use feval to evaluate the estimate for the i-th resample
   bvals(i) = feval(fname, xstar);
end
sehatT = sqrt(var(bvals));
bhat = mean(bvals)-thetahat; 

% FIND THE JACKKNIFE-AFTER-BOOTSTRAP
% Now get the boostrap estimate of the standard error of sehatT.
% Note that bvals has all of the bootstrap replicates.
% Just need to pick those out.
% Loop through all points, find the cols in bootsam that
% do not have that point in it.
% Set up storage space.
jreps = zeros(1,n);
for i = 1:n
	% Note that the columns of bootsam are the indices to the samples
	[I,J] = find(bootsam==i);	% Find all samples that have the point in it
	jacksam = setxor(J,1:B);	% Find all the rows that do not have that in it.
	if ~isempty(jacksam)
		bootrep = bvals(jacksam);	% all boot samples without x_i in it 
		jreps(i) = std(bootrep);	% Gives SEhat_B or gamma_B as in chapter
	else
		disp(['No samples for x(i) with i = ',int2str(i)])
	end
end

% get the estimate of the error in the estimate of standard error
varhatB = (n-1)/n*sum((jreps-mean(jreps)).^2);

