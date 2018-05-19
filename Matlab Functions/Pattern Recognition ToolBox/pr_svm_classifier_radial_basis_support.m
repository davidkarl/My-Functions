%RBSVC Trainable automatic radial basis Support Vector Classifier
%
%   [W,KERNEL,NU,C] = RBSVC(A)
%   [W,KERNEL,NU,C] = A*RBSVC
%
% INPUT
%   A	      Dataset
%
% OUTPUT
%   W       Mapping: Radial Basis Support Vector Classifier
%   KERNEL  Untrained mapping, representing the optimised kernel
%   NU      Resulting value for NU from NUSVC (W = NUSVC(A,KERNEL,C)
%   C       Resulting value for C (W = SVC(A,KERNEL,C)
%
% DESCRIPTION
% This routine computes a classifier by NUSVC using a radial basis kernel
% with an optimised standard deviation by REGOPTC. The resulting classifier
% W is identical to NUSVC(A,KERNEL,NU). As the kernel optimisation is based
% on internal cross-validation the dataset A should be sufficiently large.
% Moreover it is very time-consuming as the kernel optimisation needs
% about 100 calls to SVC.
%
% If any class in A has less than 20 objects, the kernel is not optimised
% by a grid search but by PKSVM, using the Parzen kernel.
%
% Note that SVC is basically a two-class classifier. The kernel may
% thereby be different for all base classifiers and is separately optimised
% for each of them.
%
% SEE ALSO (<a href="http://37steps.com/prtools">PRTools Guide</a>)
% MAPPINGS, DATASETS, PROXM, SVC, NUSVC, REGOPTC, PKSVM

% Copyright: R.P.W. Duin, r.p.w.duin@37steps.com
% Faculty EWI, Delft University of Technology
% P.O. Box 5031, 2600 GA Delft, The Netherlands

function [w,kernel,nu,c] = rbsvc(a,sig)

if nargin < 2 || isempty(sig)
	sig = NaN;
end

if nargin < 1 || isempty(a)
	
	w = prmapping(mfilename,{sig});
	
else
	
	islabtype(a,'crisp');
  
  % remove too small classes, escape in case no two classes are left
  [a,m,k,c,lablist,L,w] = cleandset(a,2); 
  if ~isempty(w), [kernel,nu,c] = deal(NaN,NaN,NaN); return; end
	a = testdatasize(a,'objects');
  
	if (c > 2)
		
    % Compute c classifiers: each class against all others.	
		w = mclassc(a,prmapping(mfilename,{sig}));	 
    w = allclass(w,lablist,L);     % complete classifier with missing classes
    
  else
    
    if numel(classuse(a,2)) < 2  % at least 2 objects per class, 2 classes
      w = onec(a);               % otherwise use onec
      [kernel,nu,c] = deal(NaN,NaN,NaN);
	
    elseif isnumeric(sig) && isnan(sig) && any(classsizes(a) < 20)
      [w,kernel,nu,c] = pksvc(a); % not sufficient data for a grid search
      
    elseif isnumeric(sig) && isnan(sig) % optimise sigma
      
		  % find upper bound
		  d = sqrt(+distm(a));
		  sigmax = min(max(d)); % max: smallest furthest neighbor distance
      % find lower bound
		  d = d + 1e100*eye(size(a,1));
		  sigmin = max(min(d)); % min: largest nearest neighbor distance
		  % call optimiser
		  defs = {1};
		  parmin_max = [sigmin,sigmax];
		  [w,kernel,nu,c] = regoptc(a,mfilename,{sig},defs,[1],parmin_max,testc([],'soft'));
		
	  else % kernel is given
      
      if ~ismapping(sig)
        kernel = proxm([],'r',sig);
      else
        kernel = sig;
      end
		  [w,J,nu,c] = nusvc(a,kernel);

	  end
    
  end
	
end

w = setname(w,'RB-SVC');
return
