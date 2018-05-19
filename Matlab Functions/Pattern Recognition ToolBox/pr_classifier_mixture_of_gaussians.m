%MOGC Trainable classifier based on Mixture of Gaussians
%
%   W = MOGC(A,N)
%   W = A*MOGC([],N,R,S);
%   W = A*MOGC(N,R,S);
%
%	INPUT
%    A   Dataset
%    N   Number of mixtures (optional; default 2)
%    R,S Regularisation parameters, 0 <= R,S <= 1, see QDC
%	OUTPUT
%
% DESCRIPTION
% For each class j in A a density estimate is made by GAUSSM, using N(j)
% mixture components. Using the class prior probabilities they are combined 
% into a single classifier W. If N is a scalar, this number is applied to 
% all classes. The relative size of the components is stored in W.DATA.PRIOR.
%
% For small class sizes or for large values of N it may be difficult or 
% impossible to find the desired number of components. This may result in
% relatively long computing times.
%
% EXAMPLES
% PREX_DENSITY
%
% SEE ALSO (<a href="http://37steps.com/prtools">PRTools Guide</a>)
% DATASETS, MAPPINGS, QDC, PLOTM, TESTC

% Copyright: R.P.W. Duin, r.p.w.duin@37steps.com
% Faculty EWI, Delft University of Technology
% P.O. Box 5031, 2600 GA Delft, The Netherlands

% $Id: mogc.m,v 1.6 2009/11/23 09:22:28 davidt Exp $

function w = mogc(varargin)

  mapname = 'MoGC';
	argin = shiftargin(varargin,'integer');
  argin = setdefaults(argin,[],[],0,0);
  
  if mapping_task(argin,'definition')
    
    w = define_mapping(argin,'untrained',mapname);
    
  elseif mapping_task(argin,'training')			% Train a mapping.
  
    [a,n,r,s] = deal(argin{:});
	  islabtype(a,'crisp','soft');
    [m,k,c] = getsize(a);
    
    if length(n) == 1
      n = repmat(n,1,c);
    end
    if isempty(n)
      q = classsizes(a);
      n = repmat(2,1,c); % default, 2 components per class
      J = find(q <= k+1);% find small classes
      n(J) = 1;          % use just a single component
    end
    if length(n) ~= c
      error('Numbers of components does not match number of classes')
    end
  
    % remove too small classes, escape in case no two classes are left
    [a,m,k,c,lablist,L,w] = cleandset(a,n,qdc); 
    if ~isempty(w), return; end
    n = n(L);
      
    % Initialize all the parameters:
    a = testdatasize(a);
    a = testdatasize(a,'features');
    
    % remove small classes by setting priors to zero
    prior = getprior(a);
    
    w = [];
    d.mean = zeros(sum(n),k);
    d.cov = zeros(k,k,sum(n));
    d.prior = zeros(1,sum(n));
    d.nlab = zeros(1,sum(n));
    d.det = zeros(1,sum(n));

    if(any(classsizes(a)<n))
      error('One or more class sizes too small for desired number of components')
    end

    % Estimate a MOG for each of the classes:
    w = [];
    n1 = 1;
    t = sprintf('Running over %i classes: ',c);
    prwaitbar(c,t);
    for j=1:c
      prwaitbar(c,j,[t num2str(j)]);
      if prior(j) > 0
        % skip components with zero probability
        b = seldat(a,j)*remclass;
        %b = setlabtype(b,'soft');
        v = gaussm(b,n(j),r,s);
        n2 = n1 + size(v.data.mean,1) -1;
        d.mean(n1:n2,:) = v.data.mean;
        d.cov(:,:,n1:n2)= v.data.cov;
        d.det(n1:n2)    = v.data.det;
        d.prior(n1:n2)  = v.data.prior*prior(j);
        d.nlab(n1:n2)   = j;
        n1 = n2+1;
      end
    end
    prwaitbar(0);
    
    if n1 <= sum(n)
      d.mean(n1:end,:) = [];
      d.cov(:,:,n1:end) = [];
      d.det(n1:end) = [];
      d.prior(n1:end)  = [];
      d.nlab(n1:end) = [];
    end
    
    w = prmapping('normal_map','trained',d,getlablist(a),k,c);
    %w = normal_map(d,getlablist(a),k,c);
    w = setname(w,mapname);
    w = allclass(w,lablist,L);  % complete classifier with missing classes
    w = setcost(w,a);
    
  end
	
return;
