%RBSTATSSVC Trainable automatic radial basis Support Vector Classifier
%
%   W = RBSTATSSVC(A)
%   W = A*RBSTATSSVC
%
% INPUT
%   A	      Dataset
%
% OUTPUT
%   W       Mapping: Radial Basis Support Vector Classifier
%
% DESCRIPTION
% This routine optimizes a radial basis kernel for STATSSVC. The kernel is
% based on PROXM. As STATSSVC is basically a two-class classifier, solving
% multi-class problems in a one-against-rest fashion, different kernels may
% be needed for every base classifier. The result W is thereby a stacked
% combiner. 
%
% If any class in A has less than 20 objects, the kernel is not optimised
% but estimated by PKSTATSSVM, using the Parzen kernel.
%
% SEE ALSO (<a href="http://37steps.com/prtools">PRTools Guide</a>)
% MAPPINGS, DATASETS, PROXM, SVC, STATSSVC, PKSTATSSVC, RBLIBSVC, RBSVC

% Copyright: R.P.W. Duin, r.p.w.duin@37steps.com
% Faculty EWI, Delft University of Technology
% P.O. Box 5031, 2600 GA Delft, The Netherlands

function w = rbstatssvc(a,sig)

  checktoolbox('stats_svmtrain');
  mapname = 'StatsRBSVM';

if nargin < 2 || isempty(sig)
	sig = NaN;
end

if nargin < 1 || isempty(a)
	
	w = prmapping(mfilename,{sig});
	
else
	
	islabtype(a,'crisp');
  
  % remove too small classes, escape in case no two classes are left
  [a,m,k,c,lablist,L,w] = cleandset(a,2); 
  if ~isempty(w), sig = NaN; return; end
	a = testdatasize(a,'objects');
  
	if (c > 2)
		
    % Compute c classifiers: each class against all others.	
		w = mclassc(a,prmapping(mfilename,{sig}));	 
    w = allclass(w,lablist,L);     % complete classifier with missing classes
    
  else
    
    if numel(classuse(a,2)) < 2  % at least 2 objects per class, 2 classes
      w   = onec(a);             % otherwise use onec
      sig = NaN;
	
    elseif isnumeric(sig) && isnan(sig) && any(classsizes(a) < 20)
      
      w = pkstatssvc(a); 
      
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
		  w = regoptc(a,mfilename,{sig},defs,[1],parmin_max,testc([],'soft'));
		
	  else % kernel is given
      
      kernel = proxm([],'r',sig);
      w = statssvc(a,kernel);

	  end
    
  end
	
end
  
w = setname(w,mapname);

return
