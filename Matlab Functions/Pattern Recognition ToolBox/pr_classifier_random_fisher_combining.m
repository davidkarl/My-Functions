%RFISHERCC  Fisher combining of randomly generated classifiers
%
%    W = RFISHERCC(A,N)
%    W = A*RFISHERCC(N)
%
% INPUT
%   A  prdataset to be used for training, M objects, C classes
%   N  number of base classifiers to be generated
%      default: M/10, <= 100.
%
% OUTPUT
%   W  trained classifier
%
% DESCRIPTION
% This routine generates a a random set of N simple classifiers, based on
% the 1-NN rule using a single, randomly selected object per class. The
% confindences (see KNNC) for the total training set A (in total N*(C-1)
% per object) are used to train a combiner using FISHERC.
%
% EXAMPLES
% a = gendatb;
% figure; scatterd(a); 
% plotc(a*rfishercc)
% 
% a = gendatm; 
% figure; scatterd(a); 
% plotc(a*rfishercc(2),'col')
%
% a = setprior(sonar,0); % make priors equal
% w1 = setname(rfishercc(10),'RFisher-10');
% w2 = setname(rfishercc(20),'RFisher-20');
% w3 = setname(rfishercc(40),'RFisher-40');
% randreset(1); % for reproducability
% e = cleval(a,{w1,w2,w3},[5,10,20,40,80],10);
% plote(e);
%
% SEE ALSO (<a href="http://37steps.com/prtools">PRTools Guide</a>)
% MAPPINGS, DATASETS, KNNC, FISHERC

% Copyright: R.P.W. Duin, r.p.w.duin@37steps.com

function w = rfishercc(varargin)

mapname = 'RandFisherCC';
argin = shiftargin(varargin,'scalar');
argin = setdefaults(argin,[],[]);
if mapping_task(argin,'definition')
  w = define_mapping(argin,'untrained',mapname);
  
elseif mapping_task(argin,'training')
  [a,n] = deal(argin{:});
  
  if isnan(n)    % optimise regularisation parameter
    defs = {1};
    parmin_max = [max(2,floor(size(a,1)/50)),ceil(size(a,1)/2)];
    w = regoptc(a,mfilename,{n},defs,[1],parmin_max,testc([],'soft'),0);
    
  else
    % remove too small classes, escape in case no two classes are left
    [a,m,k,c,lablist,L,w] = cleandset(a,1); 
    if ~isempty(w), return; end

    % default number of classifiers, <= 100
    if isempty(n)
      n = max(min(ceil(m/10),100),2);
    end
    u = a*repmat({gendat([],ones(1,c))*knnc([],1)},1,n);  
    u = u*featsel(1:c-1);
    w = a*(stacked(u)*fisherc);
    w = allclass(w,lablist,L);     % complete classifier with missing classes
    w = setname(w,mapname);
  end

else
  error('Wrong input')
end