%ADABOOSTC Computation of a combined classifier according to adaboost.
%
% [W,V,ALF] =  ADABOOSTC(A,CLASSF,N,RULE,VERBOSE);
%
% INPUT
%   A       Dataset
%   CLASSF  Untrained weak classifier
%   N       Number of classifiers to be trained, default 100
%   RULE    Combining rule (default: weighted voting)
%   VERBOSE Suppress progress report if 0 (default)
%
% OUTPUT
%   W       Combined trained classifier
%   V       Cell array of all classifiers
%           Use VC = stacked(V) for combining
%   ALF     Weights
%
% DESCRIPTION
% In total N weighted versions of the training set A are generated
% iteratevely and used for the training of the specified classifier.
% Weights, to be used for the probabilities of the objects in the training
% set to be selected, are updated according to the Adaboost rule.
%
% The generation of base classiifers may be stopped prematurely by PRTIME.
%
% The entire set of generated classifiers is given in V.
% The set of classifier weigths, according to Adaboost is returned in ALF
%
% Various aggregating possibilities can be given in 
% the final parameter rule:
% []:      WVOTEC, weighted voting.
% VOTEC    voting
% MEANC    sum rule
% AVERAGEC averaging of coeffients (for linear combiners)
% PRODC    product rule
% MAXC     maximum rule
% MINC     minimum rule
% MEDIANC  median rule
%
% REFERENCE
% Ji Zhu, Saharon Rosset, Hui Zhou and Trevor Hastie, 
% Multiclass Adaboost. A multiclass generalisation of the Adaboost 
% algorithm, based on a generalisation of the exponential loss.
% http://www-stat.stanford.edu/~hastie/Papers/samme.pdf
%
% SEE ALSO (<a href="http://37steps.com/prtools">PRTools Guide</a>)
% MAPPINGS, DATASETS, PRTIME

% Copyright: R.P.W. Duin, r.p.w.duin@37steps.com
% Faculty EWI, Delft University of Technology
% P.O. Box 5031, 2600 GA Delft, The Netherlands
% (Multiclass correction by Marcin Budka, Bournemouth Univ., UK)

%%
%
function [out,V,alf] = adaboostc(varargin)

%% INITIALISATION
argin = setdefaults(varargin,[],nmc,200,[],0);
if mapping_task(argin,'definition')
  
  out = define_mapping(argin,'untrained','Adaboost');
  
%% TRAINING
elseif mapping_task(argin,'training')
  
  [a,clasf,n,rule,verbose] = deal(argin{:});
  [m,k,c] = getsize(a);
  a = setprior(a,getprior(a));
  V = [];
  laba = getlab(a);
  u = ones(m,1)/m;			% initialise object weights
  alf = zeros(1,n);			% space for classifier weights
  isseparable = 0;          % check if we can make 0 error
  if verbose && k == 2
    figure(verbose);
    scatterd(a);
  end

  
  %% generate n classifiers
	s = sprintf('Adaboost, %i seconds, %i classifiers: ',prtime,n);
	prwaitbar(prtime,s);
  starttime  = clock;
  runtime    = 0;
  for i = 1:n
    if runtime > prtime
      n = i-1; alf = alf(1:n); 
      prwarning(2,'training stopped by PRTIME with %i classifiers',n)
      break; 
    end
		prwaitbar(prtime,runtime,[s num2str(i)]);
    try
      b = gendatw(a,u,m);           % sample training set
    catch err
      if strcmp(err.identifier,'PRTools:gendatw:SmallSet')
        prwarning(2,'training set too small: fall back to KNNC')
        out = knnc(a);
        V = NaN; alf = NaN; 
        return
      else
        rethrow(err);
      end
    end        
    b = setprior(b,getprior(a));	  % use original priors
    w = b*clasf;                    % train weak classifier
    ra = a*w;                       % test weak classifier

    if verbose && k == 2
      plotc(w,1); drawnow
    end
	
    labc = labeld(ra);
    diff = sum(labc~=laba,2)~=0;	  % objects erroneously classified
    erra = sum((diff).*u);          % weighted error on original dataset

    if (erra==0)
        isseparable = 1;
        V = w;
        break;
    end
    if (erra < (1-1/c))        % if classifier better than random guessing...
      alf(i) = 0.5*(log((1-erra)/erra) + log(c-1));
      correct = find(diff==0); % find correctly classified objects
      wrong = find(diff==1);   % find incorrectly classified objects
      u(correct) = u(correct)*exp(-alf(i));	% give them the ...
      u(wrong) = u(wrong)*exp(alf(i));	  	% proper weights
      u = u./sum(u);                        % normalise weights
    else
      alf(i) = 0;
    end
	
    if verbose
      % disp([erra alf(i) sum(alf)])
    end
    V = [V w];                       % store all classifiers
    runtime = etime(clock,starttime);
  end
  prwaitbar(0);

  %% combine and return
  if isseparable
      W = V;
      W = setname(W,['Boosted ',getname(V)]);
  else
    if isempty(rule)
        W = wvotec(V,alf);             % default is weighted combiner
    else
        W = traincc(a,V,rule);         % otherwise, use user supplied combiner
    end
  end

  if verbose > 0 && k == 2
    plotc(W,'r',3)
    ee = a*W*testc;
    title(['Error: ', num2str(ee)]);
  end
  
  out = W;

else
  error('Illegal call')
end

return
