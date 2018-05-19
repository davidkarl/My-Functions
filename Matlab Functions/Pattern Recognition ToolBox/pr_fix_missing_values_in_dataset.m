%MISVAL Fixed mapping handling the missing values in a dataset
%
%    B = MISVAL(A,VAL,K)
%    B = A*MISVAL([],VAL,K)
%    B = A*MISVAL(VAL,K)
%    N = A*MISVAL([],K)
%
% INPUT
%    A    Dataset, containing NaNs (missing values).
%    VAL  String with substitution option or value used for substitution.
%         By default all objects with missing values are removed.
%    K    Vector with indices of features that should be handled.
%         Default: all
%
% OUTPUT
%    B    Dataset with NaNs substituted
%    N    Vector with number of missing values per feature
%
% DESCRIPTION
%
% The following values for VAL are possible:
%   'remove'     remove objects (rows) that contain missing values (default)
%   'f-remove'   remove features (columns) that contain missing values
%   'mean'       fill the entries with the mean of the features
%   'c-mean'     fill the entries with the class mean of the features
%   'median'     fill the entries with the median of the features
%   'c-median'   fill the entries with the class median of the features
%   'majority'   fill the entries with the majority of the features
%   'c-majority' fill the entries with the class majority of the features
%   <value>      fill the entries with a fixed constant
%   NaN          no change: return dataset as it is
%
% Note that replacing missing values by feature means (medians or 
% majorities) is a very simple solution that might be far from optimal.
%
% SEE ALSO (<a href="http://37steps.com/prtools">PRTools Guide</a>)
% DATASETS, FEATTYPES, ISMISVAL

% Copyright: D.M.J. Tax, D.M.J.Tax@37steps.com
% Faculty EWI, Delft University of Technology
% P.O. Box 5031, 2600 GA Delft, The Netherlands

function [x,msg] = misval(varargin)

  mapname = 'Missing values';
	argin = shiftargin(varargin,{'scalar','char'});
  argin = setdefaults(argin,[],[],[]);
  
  if mapping_task(argin,'definition')
    
    x = define_mapping(argin,'fixed',mapname);
    
  elseif mapping_task(argin,'fixed execution')			% Compute mapping.
    
    [y,val,K] = deal(argin{:});
    if isempty(K)
      K = [1:size(y,2)];
    end
    x = prdataset(y(:,K));
    [m,k,c] = getsize(x);
    msg = '';
    % Where are the offenders?
    I = isnan(x);
    % If there are missing values, go:
    if any(I(:))
      if isempty(val)
        x = sum(I,1);
      else
        switch val
          case {'remove' 'delete'}
            % m sized logical index of good objects
            J = sum(I,2) == 0;
            clear x
            y = y(J,:);
            msg = 'Objects with missing values have been removed.';
          case {'f-remove' 'f-delete'}
            % k sized logical index of good features
            J = sum(I,1) == 0 ;
            clear x
            % all features logical index
            F = true(size(y,2),1);
            % make F false for features which 
            % are in subset K and have NaNs  
            F(K) = J;
            y = y(:,F);
            msg = 'Features with missing values have been removed.';
          case 'mean'
            for i=1:k
              J = ~I(:,i);
              if any(I(:,i)) %is there a missing value in this feature?
                if ~any(J)
                  error('Missing value cannot be filled: all values are NaN.');
                end
                mn = mean(x(J,i));
                x(find(I(:,i)),i) = mn;
              end
            end
            msg = 'Missing values have been replaced by the feature mean.';
          case 'c-mean'
            for j=1:c
              L = findnlab(x,j);
              for i=1:k
                J = ~I(L,i);
                if any(I(L,i)) %is there a missing value in this feature for this class?
                  if ~any(J)
                   error('Missing value cannot be filled: all values are NaN.');
                  end
                  mn = mean(x(J,i));
                  x(find(I(:,i)),i) = mn;
                end
              end
            end
            msg = 'Missing values have been replaced by the class feature mean.';
          case 'median'
            for i=1:k
              J = ~I(:,i);
              if any(I(:,i)) %is there a missing value in this feature?
                if ~any(J)
                  error('Missing value cannot be filled: all values are NaN.');
                end
                mn = median(x(J,i));
                x(find(I(:,i)),i) = mn;
              end
            end
            msg = 'Missing values have been replaced by the feature median.';
          case 'c-median'
            for j=1:c
              L = findnlab(x,j);
              for i=1:k
                J = ~I(L,i);
                if any(I(L,i)) %is there a missing value in this feature for this class?
                  if ~any(J)
                   error('Missing value cannot be filled: all values are NaN.');
                  end
                  mn = median(+x(J,i));
                  x(find(I(:,i)),i) = mn;
                end
              end
            end
            msg = 'Missing values have been replaced by the class feature median.';
          case 'majority'
            for i=1:k
              J = ~I(:,i);
              if any(I(:,i)) %is there a missing value in this feature?
                if ~any(J)
                  error('Missing value cannot be filled: all values are NaN.');
                end
                mn = majority(+x(J,i));
                x(find(I(:,i)),i) = mn;
              end
            end
            msg = 'Missing values have been replaced by the feature median.';
          case 'c-majority'
            for j=1:c
              L = findnlab(x,j);
              for i=1:k
                J = ~I(L,i);
                if any(I(L,i)) %is there a missing value in this feature for this class?
                  if ~any(J)
                   error('Missing value cannot be filled: all values are NaN.');
                  end
                  mn = majority(x(J,i));
                  x(find(I(:,i)),i) = mn;
                end
              end
            end
            msg = 'Missing values have been replaced by the class feature median.';
          otherwise
            if isstr(val)
              error('unknown option')
            end
            if ~isa(val,'double')
              error('Missing values can only be filled by scalars.');
            end
            if ~isnan(val)
              x(I) = val;
              msg = sprintf('Missing values have been replaced by %f.',val);
            else
              msg = '';
            end
        end
        
        if exist('x', 'var')
          y(:,K) = +x;
        end
        x = y;
      end
    else
      if isempty(val)
        x = zeros(1,numel(K));
      end
    end
  end
  user = getuser(x);
  if ~isempty(user) && isfield(user,'desc')
    user.desc = [user.desc ' ' msg];
    x = setuser(x,user);
  end
  
  
return

function m = majority(x)

y = unique(x(:));
h = histc(x,y);
hmax = max(h);
L = find(h == hmax);
m = median(y(L));
