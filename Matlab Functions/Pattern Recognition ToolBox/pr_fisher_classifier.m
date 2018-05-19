%FISHERC Trainable classifier: Fisher's Least Square Linear Discriminant
% 
%   W = FISHERC(A)
%   W = A*FISHERC
% 
% INPUT
%   A  Dataset
%
% OUTPUT
%   W  Fisher's linear classifier 
%
% DESCRIPTION  
% Finds the linear discriminant function between the classes in the 
% dataset A by minimizing the errors in the least square sense. This is
% a multi-class implementation using the one-against-all strategy. It
% results in a set of linear base classifiers, one for every class.
% The final result may be improved significantly by using a non-linear
% trained combiner, e.g. by calling W = A*(FISHERC*QDC([],[],1e-6);
%
% FISHERC also works for soft and  target labels.
%
% For high dimensional datasets or small sample size situations, the 
% Pseudo-Fisher procedure is used, which is based on a pseudo-inverse.
%
% This classifier, like all other non-density based classifiers, does not
% use the prior probabilities stored in the dataset A. Consequently, it
% is just for two-class problems and equal class prior probabilities 
% equivalent to LDC, which assumes normal densities with equal covariance
% matrices.
%
% Note that A*(KLMS([],N)*NMC) performs a very similar operation, but uses
% the prior probabilities to estimate the mean class covariance matrix used
% in the pre-whitening operation performed by KLMS. The reduced
% dimensionality N controls some regularisation.
% 
% REFERENCES
% 1. R.O. Duda, P.E. Hart, and D.G. Stork, Pattern classification, 2nd ed.
% John Wiley and Sons, New York, 2001.
% 2. A. Webb, Statistical Pattern Recognition, Wiley, New York, 2002.
% 3. S. Raudys and R.P.W. Duin, On expected classification error of the
% Fisher linear classifier with pseudo-inverse covariance matrix, Pattern
% Recognition Letters, vol. 19, no. 5-6, 1998, 385-392.
%
% SEE ALSO (<a href="http://37steps.com/prtools">PRTools Guide</a>)
% MAPPINGS, DATASETS, TESTC, LDC, NMC, FISHERM

% Copyright: R.P.W. Duin, r.p.w.duin@37steps.com
% Faculty EWI, Delft University of Technology
% P.O. Box 5031, 2600 GA Delft, The Netherlands

% $Id: fisherc.m,v 1.6 2010/02/08 15:31:48 duin Exp $

function W = pr_fisher_classifier(a)

  % No input arguments, return an untrained mapping.
  if (nargin < 1) || (isempty(a))
    W = prmapping(mfilename); 
    W = setname(W,'Fisher');
    return;
  end
  
  L = classuse(a,1);               % find classes with at least one object
  if numel(L) < 2                  % demand at least 2 classes
    prwarning(2,'training set too small: fall back to ONEC')
    W = onec(a);
    return
  end
  lablist = getlablist(a);
  [m,k,c] = getsize(a);
  a = selclass(a,L)*remclass;      % remove missing classes
  a = testdatasize(a); 
  
  if islabtype(a,'crisp') && c > 2 % multi-class classification
    W = mclassc(a,pr_fisher_classifier);        % combine one-against-rest classifiers
    %W = W*fishers(a*W);            % use Fisher combiner
    W = allclass(W,lablist,L);     % complete classifier with missing classes
    return
  end
  
  y = gettargets(a);
  J1 = findnlab(a,1);
  J2 = findnlab(a,2);
  y(J1,:) = y(J1,:) ./ numel(J2);
  y(J2,:) = y(J2,:) ./ numel(J1);
  if islabtype(a,'soft')
    y = invsigm(y); % better to give [0,1] targets full range
  end
  u = mean(a);    
  % Shift A to the origin. This is not significant, just increases accuracy.
  % A is extended by ones(m,1), a trick to incorporate a free weight in the 
  % hyperplane definition. 
  b = [+a-repmat(u,m,1), ones(m,1)]; 

  if (rank(b) <= k)
    % This causes Fisherc to be the Pseudo-Fisher Classifier
    prwarning(3,'The dimensionality is too large. Pseudo-Fisher is trained instead.');  
    v = prpinv(b)*y;                 
 else
    % Fisher is identical to the Mean-Square-Error solution.    
    v = b\y;               
  end

  offset = v(k+1,:) - u*v(1:k,:);   % Free weight. 
  W = affine(v(1:k,:),offset,a,getlablist(a),k,c);
  % Normalize the weights for good posterior probabilities.
  W = cnormc(W,a);                  
  W = setname(W,'Fisher');

return;



function W = fishers(a)
  % simplified version of FISHERC
 	[m,k,c] = getsize(a);
  lablist = getlablist(a);
  
  y = gettargets(a);
	u = mean(a);    
	% Shift A to the origin. This is not significant, just increases accuracy.
	% A is extended by ones(m,1), a trick to incorporate a free weight in the 
	% hyperplane definition. 
	b = [+a-repmat(u,m,1), ones(m,1)]; 

	if (rank(b) <= k)
		% This causes Fisherc to be the Pseudo-Fisher Classifier
		prwarning(3,'The dimensionality is too large. Pseudo-Fisher is trained instead.');  
		v = prpinv(b)*y;                 
	else
		% Fisher is identical to the Mean-Square-Error solution.		
		v = b\y;               
	end

	offset = v(k+1,:) - u*v(1:k,:); 	% Free weight. 
	W = affine(v(1:k,:),offset,a,lablist,k,c);
	% Normalize the weights for good posterior probabilities.
 	W = cnormc(W,a);						

return;

