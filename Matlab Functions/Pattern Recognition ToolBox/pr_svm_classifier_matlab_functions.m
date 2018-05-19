%STATSVC Stats Support Vector Classifier (Matlab Stats Toolbox)
%
%   W = STATSVC(A,KERNEL,C,OPTTYPE)
%   W = A*STATSVC(KERNEL,C,OPTTYPE)
%   D = B*W
% 
% INPUT
%   A	      A PRTools dataset used fro training
%   KERNEL  Untrained mapping to compute kernel by A*(A*KERNEL) during
%           training, or B*(A*KERNEL) during testing with dataset B.
%           Default: linear kernel (PROXM('p',1))
%   C       Regularization ('boxconstraint' in SVMTRAIN)
%   OPTTYPE Desired optimizer, 'SMO' (default), 'QP' or 'LS'.
%   B       PRTools dataset used for testing
% 
% OUTPUT
%   W       Mapping: Support Vector Classifier
%   D       PRTools dataset with classification results
% 
% DESCRIPTION
% This is the PRTools interface to the support vector classifier SVMTRAIN
% in Matlab's Stats toolbox. Like PRTools SVC it is a two-class
% discriminant that also can be used for multi-class problems by an
% internal call to MCLASSC. SVMTRAIN uses the Sequential Minimal 
% Optimization (SMO) method and thereby implements the L1 soft-margin SVM
% classifier. See SVMTRAIN for more details.
%
% Non-linear kernels have to be supplied by kernel procedures like PROXM.
% It is assumed that V = A*KERNEL generates the trained kernel and B*V the
% kernel matrix with size(B,1) rows and size(A,1) columns. Forthe radial
% basis kernel use PKSTATSSVC and RBSTATSSVC
%
% Like for all other PRTools classifiers, a new dataset B can be classified
% by D = B*W. The classifier performance can be measured by D*TESTC and the
% resulting labels by D*LABELD. D is a dataset with for every class a
% column. The values can be considered as class confidences or classifier
% condaitional posteriors.
%
% STATSSVM is basically a two-class classifier. Multiclass problems are
% internally solved using MCLASSC resulting in a base classifier per class.
% The final result may be improved significantly by using a non-linear
% trained combiner, e.g. by calling W = A*(STATSSVM*QDC([],[],1e-6);
%
% Alternative SVM classifiers in PRTools are based on SVC and LIBSVC. 
%
% SEE ALSO (<a href="http://37steps.com/prtools">PRTools Guide</a>)
% DATASETS, MAPPINGS, SVMTRAIN, SVC, LIBSVC, MCLASSC, PKSTATSSVC,
% RBSTATSSVC, QDC, TESTC, LABELD

% Copyright: R.P.W. Duin, r.p.w.duin@37steps.com

function out = statssvc(varargin)

  checktoolbox('stats_svmtrain');
  mapname = 'StatsSVM';
  argin = shiftargin(varargin,'prmapping');
  argin = setdefaults(argin,[],proxm([],'p',1),1,'SMO');
  
  if mapping_task(argin,'definition')
    
   out = define_mapping(argin,'untrained',mapname);
   
  else
    [a,kernel,C,opttype] = deal(argin{:});
    
    if ~(ismapping(kernel) && istrained(kernel)) % training
      isdataset(a);
      islabtype(a,'crisp');
      a = testdatasize(a,'objects');
    
      % remove too small classes, escape in case no two classes are left
      [a,m,k,c,lablist,L,out] = cleandset(a,1); 
      if ~isempty(out), return; end
      
      if c > 2                        % solve multi-class case by recursion
        u = feval(mfilename,[],kernel,C,opttype);
        out = mclassc(a,u);           % concatenation of one-against-rest
        out = allclass(out,lablist,L);% handle with missing classes
      else                            % two class case
        labels = getlabels(a);  
        ismapping(kernel);
        isuntrained(kernel);
        prkernel(kernel);          % make kernel mapping known to prkernel
        
        pp = prrmpath('stats','svmtrain');      % check / correct path
        finishup = onCleanup(@() addpath(pp));  % restore afterwards
        if strcmpi(opttype,'SMO')
          ss = svmtrain(+a,labels,'kernel_function',@prkernel, ...
               'boxconstraint',C);
        elseif strcmpi(opttype,'QP')
          ss = svmtrain(+a,labels,'kernel_function',@prkernel, ...
               'method','QP', 'boxconstraint',C);
        elseif strcmpi(opttype,'LS')
          ss = svmtrain(+a,labels,'kernel_function',@prkernel, ...
               'method','LS', 'boxconstraint',C);
        else
          error('Unknown optimizer')
        end
        
        out = trained_mapping(a,{ss,kernel},getsize(a,3));
        out = cnormc(out,a);       % normalise outputs for confidences
        out = setname(out,mapname);
      end
    
    else                           % evaluation
      w = kernel;                  % trained classifier
      [ss,kernel] = getdata(w);    % get datafrom training
      ismapping(kernel);
      % apply shift and scaling performed by svmtrain to testdata
      a = a*cmapm(-ss.ScaleData.shift,'shift');
      a = a*cmapm(1./ss.ScaleData.scaleFactor,'scale');
      %a = a*cmapm(ss.ScaleData.shift,'shift');
      % compute kernel for test data
      out = a*(ss.SupportVectors*kernel);
      out = out*ss.Alpha;                       % apply weights
      out = out+repmat(ss.Bias,size(out,1),1);  % apply bias
      out = setdat(a,[out -out],w);             % two-class result
    end
    
  end
  
return
  



        
