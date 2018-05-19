%NMSC Nearest Mean Scaled Classifier
% 
% 	W = NMSC(A)
%   W = A*NMSC
% 
% INPUT
%   A   Trainign dataset
%
% OUTPUT
%   W   Nearest Mean Scaled Classifier mapping
%
% DESCRIPTION
% Computation of the linear discriminant for the classes in the dataset A
% assuming normal distributions with zero covariances and equal class variances. 
% The use of soft labels is supported.
%
% The difference with NMC is that NMSC is based on an assumption of normal
% distributions and thereby automatically scales the features and is
% sensitive to class priors. NMC is a plain nearest mean classifier that is
% feature scaling sensitive and unsensitive to class priors.
%
% As NMSC is a linear classifier, a non-linear combiner might give an
% improvement in multi-dimensional problems, e.g. by 
% W = A*(NMC*QDC([],[],1e-6)).
% 
% SEE ALSO (<a href="http://37steps.com/prtools">PRTools Guide</a>)
% DATASETS, MAPPINGS, NMC, LDC ,FISHERC, QDC, UDC 

% Copyright: R.P.W. Duin, duin@ph.tn.tudelft.nl
% Faculty of Applied Physics, Delft University of Technology
% P.O. Box 5046, 2600 GA Delft, The Netherlands

% $Id: nmsc.m,v 1.7 2008/01/25 10:20:07 duin Exp $

function w = pr_nearest_mean_linear_scaled_classifier(varargin)
  
	mapname = 'S-NearestMean';
  argin = setdefaults(varargin,[]);
  
  if mapping_task(argin,'definition')
    w = define_mapping(argin,'untrained',mapname);
    
  elseif mapping_task(argin,'training')			% Train a mapping.

    a = argin;
    islabtype(a,'crisp','soft');
  
      % remove too small classes, escape in case no two classes are left
      [a,m,k,c,lablist,L,w] = cleandset(a,1); 
      if ~isempty(w), return; end
      
    p = getprior(a);
    [U,GG] = meancov(a);

    % All class covariance matrices are assumed to be diagonal. They are
    % weighted by the priors, unlike the standard nearest mean classifier (NMC).
    
    G = zeros(c,k);
    for j = 1:c
      G(j,:) = diag(GG(:,:,j))';
    end
    G = p*G;

    % The two-class case is special, as it can be conveniently stored as an
    % affine mapping.

    if (c == 2)
      ua = +U(1,:); ub = +U(2,:);
      R = G*(ua - ub)';
      R = ((ua - ub)./G)';
      offset = ((ub./G)*ub' - (ua./G)*ua')/2 + log(p(1)/p(2));
      w = affine(R,offset,a,getlablist(a),k,c); 
      w = cnormc(w,a);
    else
      pars.mean      = +U; 
      pars.cov       = G; 
      pars.prior     = p;
      w = normal_map(pars,getlablist(a),k,c);
    end

    w = setname(w,mapname);
    w = allclass(w,lablist,L);     % complete classifier with missing classes
    w = setcost(w,a);
  
  end

return
