%CLEANDSET Clean dataset for small class size behavior of classifiers
%
%    [B,M,K,C,LABLIST,L,W] = CLEANDSET(A,N,U)
%
% INPUT
%   A        Dataset
%   N        Minimum desired class size, default 1
%   U        Untrained fallback classifier, default ONEC
%
% OUTPUT
%   B        Dataset with small and empty classes removed
%   M        Number of objects in B
%   K        Feature size of B
%   C        Number of classes in B
%   LABLIST  Label list of A
%   L        Classes of A still availiable in B
%   W        Trained fallback classifier
%
% DESCRIPTION
% This routine serves three purposes:
% - It summarises a number of statements in the training parts of a
%   classifier in orer to make the source more readable.
% - Removal of small classes.
% - In case B does not contain at least two classes of the desired sample
%   size, the fallback classifier U is trained by A and returned in W.
%
% This routine takes facilitates the handlin of imcomplete training sets,
% together with the support routines ONEC, ALLCLASS and, CLASSUSE.
%
% SEE ALSO (<a href="http://37steps.com/prtools">PRTools Guide</a>)
% DATASETS, MAPPINGS, ONEC, ALLCLASS, CLASSUSE

% Copyright: R.P.W. Duin, r.p.w.duin@37steps.com


function [a,m,k,c,lablist,L,varargout] = pr_clean_dataset(varargin)

[a,n,u] = setdefaults(varargin,[],1,onec);
isdataset(a);
varargout = cell(1,nargout-6);
if islabtype(a,'crisp')
  L = classuse(a,n);
  if numel(L) < 2
    prwarning(2,['training set too small: fall back to ' upper(getmapping_file(u))])
    [varargout{:}] = a*u;
  else
    varargout{1} = [];
  end
  lablist = getlablist(a);
  a = seldat(a,L);
  [m,k,c] = getsize(a);
else
  [m,k,c] = getsize(a);
  L = [1:c];
  lablist = getlablist(a);
  varargout{1} = [];
end