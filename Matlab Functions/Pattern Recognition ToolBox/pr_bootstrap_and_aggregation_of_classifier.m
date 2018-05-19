%BAGGINGC Bootstrapping and aggregation of classifiers
% 
%    W = BAGGINGC (A,CLASSF,N,ACLASSF,T)
% 
% INPUT
%   A         Training dataset.
%   CLASSF    The base classifier (default: NMC)
%   N         Number of base classifiers to train (default: 100)
%   ACLASSF   Aggregating classifier (default: MAXC).
%   T         Tuning set on which ACLASSF is trained (default: [], meaning use A)
%
% OUTPUT
%    W        A combined classifier (if ACLASSF given) or a stacked
%             classifier (if ACLASSF []).
%
% DESCRIPTION
% Computation of a stabilised version of a classifier by bootstrapping and
% aggregation ('bagging'). In total N bootstrapped versions of the dataset A
% are generated and used for training of the untrained classifier CLASSF.
% Aggregation is done using the combining classifier specified in ACLASSF.
% If ACLASSF is a trainable classifier it is trained by the tuning dataset
% T, if given; else A is used for training. The default aggregating classifier
% ACLASSF is MAXC. Default base classifier CLASSF is NMC.
%
% In case the aggregating classifier ACLASSF is WVOTEC the weights for the
% voting are derived from the apparent errors based on the bootstrapped
% versions of the training set A.
%
% In multi-class problems another way of combining might be of interest:
% W = A*(BAGGINGC*QDC([],[],1e-6)).
%
% REFERENCE
% L.Beiman, Bagging Predictors, Machine Learning, 1996.
%
% SEE ALSO (<a href="http://37steps.com/prtools">PRTools Guide</a>)
% DATASETS, MAPPINGS, NMC, MAXC

% Copyright: R.P.W. Duin, duin@ph.tn.tudelft.nl
% Faculty of Applied Sciences, Delft University of Technology
% P.O. Box 5046, 2600 GA Delft, The Netherlands

% $Id: baggingc.m,v 1.3 2010/06/01 08:47:05 duin Exp $

function w = pr_bootstrap_and_aggregation_of_classifier(varargin)


  mapname = 'Bagging';
  argin = setdefaults(varargin,[],nmc,100,[],[]);
  
  if mapping_task(argin,'definition')
    
    w = define_mapping(argin,'untrained',mapname);
    
  else

    [a,clasf,n,rule,t] = deal(argin{:});
    iscomdset(a,t); % test compatibility training and tuning set

    % Concatenate N classifiers on bootstrap samples (100%) taken
    % from the training set.

    w = [];
    s = sprintf('Generation of %i base classifiers: ',n);
    v = zeros(1,n);
    w = cell(1,n);
    prwaitbar(prtime,s);
    starttime = clock;
    runtime = 0;
    for i = 1:n
      prwaitbar(prtime,runtime,[s num2str(i)]);
      if runtime > prtime
        n = i-1;
        w = w(1:n);
        v = v(1:n);
        prwarning(2,'Generation of base classifiers stopped by PRTIME at %i classifiers',i-1);
        break;
      end
      aa   = gendat(a);
      w{i} = aa*clasf;
      prob = testc(aa*w{i});
      v(i) = 0.5*(log((1-prob)/(prob+1e-10))); % classifier weights
      % w = [w gendat(a)*clasf]; 
      runtime = etime(clock,starttime);
    end
    prwaitbar(0);

    % If no aggregating classifier is given, just return the N classifiers...

    if (~isempty(rule))

      % ... otherwise, train the aggregating classifier on the train or
      % tuning set.
      w = stacked(w);
      if  strcmp(getmapping_file(rule),'wvotec')==1,
        w = wvotec(w,v);
      else
        w = traincc(a,w,rule);
      end
      w = setname(w,mapname);
    end

    w = setcost(w,a);
    
  end
	
return
