%GENMDAT Generate multidimensional dataset
%
%   A = GENMDAT(COMMAND,K,N)
%
% INPUT  
%   COMMAND Name of command to be used for generating the data, 
%           default 'gendatb'
%   K       Desired dimensionality, default K = 5.
%   N       Desired total number of objects, or a vector with desired number
%           per class. Default N = 100.
%
% OUTPUT
%   A       Dataset with N or SUM(N) number of objects and K features.
%
% DESCRIPTION
% The base routine COMMAND (e.g. GENDATB) is called a number of times and
% resulting datasets are concatenated, such that a K-dimensional result is
% obtained. The base routine is called as COMMAND(N). Possible commands are
% GENDATB, GENDATC, GENDATD, GENDATH, GENDATL, GENDATM, GENDATMM, GENDATS
%
% A possible alternative for this routine is GENDATV
%
% SEE ALSO (<a href="http://37steps.com/prtools">PRTools Guide</a>)
% DATASETS, GENDATB, GENDATC, GENDATD, GENDATH, GENDATL, GENDATM, GENDATMM, 
% GENDATS, GENDATV

% Copyright: R.P.W. Duin, r.p.w.duin@37steps.com


function a = pr_generate_dataset_multi_dimensional_using_generation_function(varargin)

[cmd,k,n] = setdefaults(varargin,'gendatb',5,100);

a = feval(cmd,n);
kk = size(a,2);
k0 = kk;
if k > k0
  a = [a zeros(size(a,1),k-kk)];
  while k > k0
    x = feval(cmd,n);
    a(:,k0+1:min(k,k0+kk)) = x(:,1:min(k,k0+kk)-k0);
    k0 = k0 + kk;
  end
end
a = a(:,1:k);
  