%GENDATM Generation of multimodal multi-class 2-D data
% 
% 	A = GENDATM(N)
% 
% INPUT
%   N   Vector with 4 class sizes (default: all 50)
%
% OUTPUT
%   A   Dataset
%
% DESCRIPTION
% Generation of N samples in 4 classes of 2 dimensionally distributed data
% vectors. Classes have equal prior probabilities. If N is a vector of
% sizes, exactly N(I) objects are generated for class I, I = 1..4.
%
% This dataset is based on combining classes of the 8-class problem GENDATM. 
% 
% SEE ALSO (<a href="http://37steps.com/prtools">PRTools Guide</a>)
% DATASETS, PRDATASETS, GENDATM

% Copyright: R.P.W. Duin, r.p.w.duin@37steps.com

function a = pr_generate_dataset_4_classes_2_dimensional(varargin)

p = [0.25 0.25 0.25 0.25];
n = setdefaults(varargin,50);
if numel(n) == 1
  n = genclass(1000,p);
elseif (numel(n) ~= 4) || any(n < 0)
  error('Number of elements should be a vector of 4 non-negative integers');
end
n = round(n);

m1 = genclass(n(1),[0.5 0.5]);
m2 = genclass(n(2),[0.5 0.5]);
m3 = genclass(n(3),[0.5 0.5]);
m4 = genclass(n(4),[0.5 0.5]);

a = gendatm([m1(1) m2(1) m3(1) m4(1) m2(2) m3(2) m4(2) m1(2)]);
L = [1 2 3 4 2 3 4 1];
nlab = getnlab(a);
nlab = L(nlab);
a = setnlab(a,nlab);
a = a*remclass;
a = setprior(a,p);
a = setname(a,'Multi-modal multi-class problem');