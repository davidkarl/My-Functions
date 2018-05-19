%FEATTYPES Determine feature types
%
%   [R,J,C,M,E]  = feattypes(A)
%   [R,J,C,M,E]  = A*feattypes
%   [L1,L2, ...] = feattypes(A,T1,T2, ...]
%   [L1,L2, ...] = A*feattypes(T1,T2, ...]
% INPUT
%   A           Dataset
%   T1,T2, ...  Character strings specifying desired feature types, 
%               possible values 'R','J','J2','J3','C','C2','C3','M','E'.
%
% OUTPUT
%   R            Indices of features with real values
%   J            Indices of integer features (J = [J2 J3])
%   C            Indices of categorical features (C = [C2 C3])
%   L1,L2 ...    Indices of features corresponding to T1, T2, ...
%
% DESCRIPTION
% This function sorts features in various types. This might be useful for
% applying different classifiers for different feature types. Note that
% missing values are coded in PRTools by a NaN. A simple procedure to
% handle them is MISVAL.
% - 'R'   Real valued features
% - 'J'   Integer valued features
% - 'J2'  Binary (two-valued integer) features (not necessarily 0/1)
% - 'J3'  Integer features with more than two different valuers
% - 'C'   Categorical features
% - 'C2'  Categorical features with just two different values
% - 'C3'  Categorical features with more than two different values
% - 'M'   Features with only NaNs (all missing values)
% - 'E'   Features with the same value for all objects
% - 'CM'  Categorical features having the same values or NaN for all objects
% - 'JM'  Integer features having the same value or NaN for all objects
% - 'RM'  Real valued features having the same value or NaN for all objects
%
% SEE ALSO <a href="http://37steps.com/prtools">PRTools Guide</a>
% DATASETS, MAPPINGS, FEATSEL, SETFEATDOM. MISVAL, CAT2DSET, CELL2DSET

% Copyright: R.P.W. Duin, r.p.w.duin@37steps.com

function varargout = pr_feature_selection_by_type(a,varargin)

if nargin < 1 || isempty(a)
  varargout = {prmapping(mfilename,'fixed',varargin{:})};
  return
end

isdataset(a);

if isempty(varargin)
  varargin = {'R','J','C','M','E'};
end

if nargout > numel(varargin)
  error('More outputs requested than available')
end

varout = cell(1,numel(varargin));
featdom = getfeatdom(a,1:size(a,2));
type   = cell(1,size(a,2));
for j=1:numel(type)
  x = +a(:,j);
  L = isnan(x);
  if ischar(featdom{j})
    if size(featdom{j},1) == 1
      type{j} = 'C1';
    elseif size(featdom{j},1) == 2
      type{j} = 'C2';
    else
      type{j} = 'C3';
    end
    if numel(unique(x)) == 1
      type{j} = 'E';
    end
    if all(L)
      type{j} = 'C0';
    elseif any(L)
      if numel(unique(x(~L))) == 1
        type{j} = 'CM';
      end
    end
  else
    y = x(~L);
    u = unique(y);
    if all(L)
      type(j) = 'M';
    elseif all(round(y) == y)
      if numel(u) == 1
        if all(~L)
          type{j} = 'J1';
        else
          type{j} = 'JM';
        end
      elseif numel(u) == 2
        type{j} = 'J2';
      else
        type{j} = 'J3';
      end
    else
      if numel(u) == 1
        if all(~L)
          type{j} = 'S';
        else
          type{j} = 'SM';
        end
      else
        type{j} = 'R';
      end
    end
  end
end

X = repmat({[]},1,13);
[R,J,J2,J3,C,C2,C3,M,E,C0,CM,JM,RM] = deal(X{:});
for j=1:numel(varargin)
  switch upper(varargin{j})
    case 'R'
      R = find(strcmp(type,'R'));
      varout{j} = R;
    case 'C'
      C = find(strcmp(type,'C0') | strcmp(type,'C1') | strcmp(type,'C2') | ...
               strcmp(type,'C3') | strcmp(type,'E') | strcmp(type,'CM'));
      varout{j} = C;
    case 'C0'
      C0 = find(strcmp(type,'C0'))
    case 'C2'
      C2 = find(strcmp(type,'C2'));
      varout{j} = C2;
    case 'C3'
      C3 = find(strcmp(type,'C3'));
      varout{j} = C3;
    case 'J'
      J = find(strcmp(type,'J1') | strcmp(type,'J2') | ...
               strcmp(type,'J3') | strcmp(type,'JM'));
      varout{j} = J;
    case 'J2'
      J2 = find(strcmp(type,'J2'));
      varout{j} = J2;
    case 'J3'
      J3 = find(strcmp(type,'J3'));
      varout{j} = J3;
    case 'M'
      M = find(strcmp(type,'C0') | strcmp(type,'M'));
      varout{j} = M;
    case 'E'
      E = find(strcmp(type,'E') | strcmp(type,'J1') | strcmp(type,'S'));
      varout{j} = E;
    case 'CM'
      CM = find(strcmp(type,'CM'));
      varout{j} = CM;
    case 'JM'
      JM = find(strcmp(type,'JM'));
      varout{j} = JM;
    case 'RM'
      RM = find(strcmp(type,'RM'));
      varout{j} = RM;
    otherwise
      error('Unknown feature type requested');
  end
end

if nargout > 0 || nargin > 1
  varargout = varout;
else
  fprintf('\nFeature types found:\n');
  fprintf('--------------------\n');
  printfeat(' Real:         ',R);
  printfeat(' Categorical:  ',C);
  printfeat(' Integer:      ',J);
  printfeat(' All missing:  ',M);
  printfeat(' All equal:    ',E);
  fprintf('\n');
end

return

function printfeat(desc,R)
if ~isempty(R)
  fprintf(desc);
  if isempty(R)
    fprintf('none\n');
  else
    fprintf(' %i',R);
    fprintf('\n');
  end
end

      

      
    
