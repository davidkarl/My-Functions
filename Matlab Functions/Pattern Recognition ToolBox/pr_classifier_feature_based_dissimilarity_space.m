%FDSC Trainable Feature based Dissimilarity Space Classifier 
%
%   W = FDSC(A,R,FEATMAP,TYPE,P,CLASSF)
%   W = A*FDSC([],R,FEATMAP,TYPE,P,CLASSF)
%   D = X*W
%
% INPUT
%   A       Dateset used for training
%   R       Dataset used for representation
%           or a fraction of A to be used for this.
%           Default: R = A.
%   FEATMAP Preprocessing in feature space (e.g. SCALEM)
%           Default: no preprocessing.
%   TYPE    Dissimilarity rule, see PROXM
%           Default 'DISTANCE'.
%   P       Parameter for dissimilarity rule, see PROXM
%           Default P = 1.
%   CLASSF  Classifier used in dissimilarity space
%           Default LIBSVC([],[],100)
%   X       Test set
%
% OUTPUT
%   W       Resulting, trained feature space classifier
%   D       Classification matrix
%
% DESCRIPTION
% This routine builds a classifier in feature space based on a 
% dissimilarity representation defined by the representation set R
% and the dissimilarities found by A*FEATMAP*PROXM(R*FEATMAP,TYPE,P).
% FEATMAP is a preprocessing in feature space, e.g. scaling
% (SCALEM([],'variance') or pre-whitening (KLMS).
%
% R can either be explicitely given, or by a fraction of A. In the
% latter case the part of A that is randomly generated to create the
% representation set R is excluded from the training set.
%
% New objects in feature space can be classified by D = B*W or by
% D = PRMAP(B,W). Labels can be found by LAB = D*LABELD or LAB = LABELD(D).
%
% EXAMPLE
% a = gendatb([100 100]);    % training set of 200 objects
% r = gendatb([10 10]);      % representation set of 20 objects
% w = fdsc(a,r);             % compute classifier
% scatterd(a);               % scatterplot of trainingset
% hold on; scatterd(r,'ko'); % add representation set to scatterplot
% plotc(w);                  % plot classifier
%
% SEE ALSO (<a href="http://37steps.com/prtools">PRTools Guide</a>)
% DATASETS, MAPPINGS, SCALEM, KLMS, PROXM, LABELD, KERNELC

% Copyright: R.P.W. Duin, r.p.w.duin@37steps.com
% Faculty EWI, Delft University of Technology
% P.O. Box 5031, 2600 GA Delft, The Netherlands

function w = fdsc(varargin)

if nargin > 0 && ismapping(varargin{1})
  defclassf = varargin{1};
  varargin{1} = [];
else
  defclassf = libsvc([],[],100);
end
argin = setdefaults(varargin,[],[],[],'distance',1,defclassf);
[a,r,featmap,type,p,classf] = deal(argin{:});
if mapping_task(argin,'definition')
	w1 = prmapping(mfilename,'untrained',{r,featmap,type,p,classf}); 
	w = setname(w1,['DisSpace-',getname(classf)]);
	return
end

if isdataset(a) || isdatafile(a)
  if numel(classuse(a,1)) < 2 % at least 1 object per class, 2 classes
    prwarning(2,'training set too small: fall back to ONEC')
    w = onec(a);
    return
  end
	if isempty(featmap)
		v = scalem(a); % shift mean to the origin
	elseif isuntrained(featmap)
		v = a*featmap;
	elseif ismapping(featmap)
		v = featmap;
	else
		error('Feature mapping expected')
	end
	if isempty(r)
		r = a;
	elseif is_scalar(r)
		[r,a] = gendat(a,r);
	end
	w = proxm(r*v,type,p);
	if ~isuntrained(classf)
		error('Untrained classifier expected')
	end
	b = a*v*w;
	%b = testdatasize(b);
	n = disnorm(b);
	u = b*n*classf;
	w = v*w*n*u;
else
	error('Unexpected input')
end
	

	

