%RANDOMFORESTC Breiman's random forest
%
%   W = RANDOMFORESTC(A,L,N)
%   W = A*RANDOMFORESTC([],L,N)
%
% INPUT
%   A       Dateset used for training
%   L       Number of decision trees to be generated (default 200)
%   N       Size of feature subsets to be used (default 1)
%
% OUTPUT
%   W       Resulting, trained feature space classifier
%
% DESCRIPTION
% Train a decision forest on A, using L decision trees, each trained on
% a bootstrapped version of dataset A. Each decison tree is using random
% feature subsets of size N in each node.  When N=0, no feature subsets
% are used.
%
% The generation of trees might be slow. It might be stopped by PRTIME
% before L trees are constructed. Set PRTIME larger if desired.
%
% REFERENCES
% [1] L. Breiman, Random Forests, Machine learning, vol. 45 (1), 5-32, 2001
%
% SEE ALSO (<a href="http://37steps.com/prtools">PRTools Guide</a>)
% DATASETS, MAPPINGS, DTC, PRTIME

% Copyright: D.M.J. Tax, D.M.J.Tax@37steps.com
% Faculty EWI, Delft University of Technology
% P.O. Box 5031, 2600 GA Delft, The Netherlands

function out = randomforestc(varargin)

mapname = 'RandForest';
argin = setdefaults(varargin,[],200,1);
if mapping_task(argin,'definition')
  
  out = define_mapping(argin,'untrained',mapname);
  
elseif mapping_task(argin,'training')
  
  [a,L,featsubset] = deal(argin{:});
  if numel(classuse(a,1)) < 2  % at least 2 objects per class, 2 classes
    prwarning(2,'training set too small: fall back to ONEC')
    out = onec(a);
    return
  end
  opt = [];
  [n,dim,opt.K] = getsize(a);
  opt.featsubset = featsubset;
  v = cell(L,1);
  warn = false;
	s = sprintf('random forest, %i seconds, %i trees: ',prtime,L);
	prwaitbar(prtime,s);
  starttime  = clock;
  runtime    = 0;
  for i=1:L
    if runtime > prtime
      prwarning(2,'random forest training stopped by PRTIME with %i trees',i-1);
      L = i-1; v = v(1:L);
      break;
    end
		prwaitbar(prtime,runtime,[s num2str(i)]);
    [x,z] = gendat(a);
    if exist('decisiontree','file')==3
      v{i} = decisiontree(+x,getnlab(x),opt.K,opt.featsubset);
    else
      if ~warn
        prwarning(2,'No compiled decision tree found, using the slower Matlab implementation.');
        warn = true;
      end
      v{i} = tree_train(+x,getnlab(x),opt);
    end
    runtime = etime(clock,starttime);
  end
  prwaitbar(0);
  out = trained_classifier(a,v);
  out = setname(out,mapname);
  
elseif mapping_task(argin,'execution')
  
  [a,w] = deal(argin{1:2}); 
  v = getdata(w);
  n = size(a,1);  % nr objects
  K = size(w,2);  % nr of classes
  nrv = length(v); % nr of trees
    out = zeros(n,K);
    if exist('decisiontree','file')==3
      for j=1:nrv
        I = decisiontree(v{j},+a);
        out = out + accumarray([(1:n)' I],ones(n,1),[n K]);
      end
    else
      % the old fashioned slow Matlab code
      t = sprintf('Evaluating randomforest by %i objects: ',n);
      prwaitbar(n,t);
      for i=1:n
        prwaitbar(n,i,[t num2str(i)]);
        x = +a(i,:);
        for j=1:nrv
          I = tree_eval(v{j},x);
          out(i,I) = out(i,I)+1;
        end
      end
      prwaitbar(0);
      out = out./repmat(sum(out,2),1,K);
    end
    out = setdat(a,out,w);
    
else
  error('Illegal call')
end

return

%    out = tree_eval(w,x)
%
function out = tree_eval(w,x)

n = size(x,1);
out = zeros(n,1);

for i=1:n

  v=w;
  % if the first split is already solving everything (1 obj. per class)
  if isa(v,'double')
    out(i,1) = v;
  end
  while (out(i,1)==0)
    if (x(i,v.bestf)<v.bestt)
      v = v.l;
    else
      v = v.r;
    end
    if isa(v,'double')
      out(i,1) = v;
    end
  end
end

%
%    w = tree_train(x,y,opt)
%
function w = tree_train(x,y,opt)

% how good are we in this node?
err = tree_gini(y,opt.K);
if (err==0)

  w = y(1); % just predict this label

else
  % we split further
  n = size(x,1);

  % optionally, choose only from a subset
  if (opt.featsubset>0)
    fss = randperm(size(x,2));
    fss = fss(1:opt.featsubset);
  else
    fss = 1:size(x,2);
  end

  % check each feature separately:
  besterr = inf; bestf = []; bestt = []; bestj = []; bestI = [];
  for i=fss
    % sort the data along feature i:
    [xi,I] = sort(x(:,i)); yi = y(I);
    % run over all possible splits:
    for j=1:n-1
      % compute the gini
      err = j*tree_gini(yi(1:j),opt.K) + (n-j)*tree_gini(yi(j+1:n),opt.K);
      % and see if it is better than before.
      if (err<besterr)
        besterr = err;
        bestf = i;
        bestj = j;
        bestt = mean(xi(j:j+1));
        bestI = I;
      end
    end
  end

  % store
  w.bestf = bestf;
  w.bestt = bestt;
  %  now find the children:
  w.l = tree_train(x(bestI(1:bestj),:),y(bestI(1:bestj)),opt);
  w.r = tree_train(x(bestI(bestj+1:end),:),y(bestI(bestj+1:end)),opt);
end
  
function g = tree_gini(y,K)

out = zeros(1,K);
for k=1:K
  out(k) = mean(y==k);
end

g = out*(1-out)';

