%RBSVC Trainable automatic radial basis Support Vector Classifier by LIBSVM
%
%   [W,KERNEL,NU] = RBLIBSVC(A)
%   [W,KERNEL,NU] = A*RBLIBSVC
%
% INPUT
%   A	      Dataset
%
% OUTPUT
%   W       Mapping: Radial Basis Support Vector Classifier
%   KERNEL  Untrained mapping, representing the optimised kernel
%   NU      Resulting value for NU from NUSVC
%
% DESCRIPTION
% This routine computes a classifier by NULIBSVC using a radial basis kernel
% with an optimised standard deviation by REGOPTC. The resulting classifier
% W is identical to NULIBSVC(A,KERNEL,NU). As the kernel optimisation is based
% on internal cross-validation the dataset A should be sufficiently large.
% Moreover it is very time-consuming as the kernel optimisation needs
% about 100 calls to LIBSVC.
%
% Note that LIBSVC is basically a multi-class classifier. The kernel
% is thereby the same for all classes. Use MCLASSC to find a multi-class
% classifier composed by two-class classifiers with possibly different
% kernels.
%
% SEE ALSO (<a href="http://37steps.com/prtools">PRTools Guide</a>)
% MAPPINGS, DATASETS, PROXM, LIBSVC, NULIBSVC, REGOPTC

% Copyright: R.P.W. Duin, r.p.w.duin@37steps.com
% Faculty EWI, Delft University of Technology
% P.O. Box 5031, 2600 GA Delft, The Netherlands

function [w,kernel,nu] = rblibsvc(a,sig)

if nargin < 2 | isempty(sig)
	sig = NaN;
end

if nargin < 1 | isempty(a)
	
	w = prmapping(mfilename,{sig});
	
else
	
	islabtype(a,'crisp');
	a = testdatasize(a,'objects');
  c = getsize(a,3);
	islabtype(a,'crisp');
  
  if any(classsizes(a) < 20) && isnan(sig)
    [w,kernel,nu] = pklibsvc(a);
    prwarning(2,'Training set too small, kernel estimated by PKSVC');
  	
  elseif isnan(sig) % optimise sigma
      
	  % find upper bound
	  d = sqrt(+distm(a));
	  sigmax = min(max(d)); % max: smallest furthest neighbor distance
    % find lower bound
	  d = d + 1e100*eye(size(a,1));
	  sigmin = max(min(d)); % min: largest nearest neighbor distance
	  % call optimiser
	  defs = {1};
	  parmin_max = [sigmin,sigmax];
	  [w,kernel,nu] = regoptc(a,mfilename,{sig},defs,[1],parmin_max,testc([],'soft'));
		
  else % kernel is given
		
	  kernel = proxm([],'r',sig);
	  [w,J,nu] = nulibsvc(a,kernel);

	end
    	
end

w = setname(w,'RB-LIBSVM');
return
