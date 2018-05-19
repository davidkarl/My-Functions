%NUSVC Trainable classifier: Support Vector Machine, nu-algorithme
% 
% 	[W,J,NU] = NUSVC(A,KERNEL,NU)
%   [W,J,NU] = A*NUSVC([],KERNEL,NU)
%   [W,J,NU] = A*NUSVC(KERNEL,NU)
%
% INPUT
%   A	      Dataset
%   KERNEL  - Untrained mapping to compute kernel by A*(A*KERNEL) during
%             training, or B*(A*KERNEL) during testing with dataset B.
%           - String to compute kernel matrices by FEVAL(KERNEL,B,A)
%           Default: linear kernel (PROXM([],'p',1));
%   NU      Regularisation parameter (0 < NU < 1): expected fraction of SV 
%           (optional; default: max(leave-one-out 1_NN error,0.01))
%
% OUTPUT
%   W       Mapping: Support Vector Classifier
%   J       Object indices of support objects		
%   NU      Actual nu_value used
%
% DESCRIPTION
% Optimises a support vector classifier for the dataset A by quadratic 
% programming. The difference with the standard SVC routine is the use and
% interpretation of the regularisation parameter NU. It is an upperbound
% for the expected classification error. By default NU is estimated by the
% leave-one-error of the 1_NN rule. For NU = NaN an automatic optimisation 
% is performed using REGOPTC. 
%
% If KERNEL = 0 it is assumed that A is already the kernelmatrix (square).
% In this case also a kernel matrix B should be supplied at evaluation by
% B*W or PRMAP(B,W).
%
% There are several ways to define KERNEL, e.g. PROXM([],'r',1) for a
% radial basis kernel or by USERKERNEL for a user defined kernel.
%
% SVC is basically a two-class classifier. Multi-class problems are solved
% in a one-against-rest fashion by MCLASSC. The resulting base-classifiers
% are combined by the maximum confidence rule. A better, non-linear
% combiner might be QDC, e.g. W = A*(SVC*QDC([],[],1e-6))
%
% SEE ALSO (<a href="http://37steps.com/prtools">PRTools Guide</a>)
% MAPPINGS, DATASETS, SVC, NUSVO, PROXM, USERKERNEL, REGOPTC, MCLASSC, QDC

% Copyright: S.Verzakov, s.verzakov@ewi.tudelft.nl 
% Based on SVC by: D. de Ridder, D. Tax, R.P.W. Duin, r.p.w.duin@37steps.com
% Faculty EWI, Delft University of Technology
% P.O. Box 5031, 2600 GA Delft, The Netherlands
  
% $Id: nusvc.m,v 1.6 2010/06/25 09:50:40 duin Exp $

function [W,J,nu,C,alginf] = nusvc(varargin)


  mapname = 'nuSVC';
  argin = shiftargin(varargin,{'prmapping','char','cell'});
  argin = setdefaults(argin,[],proxm([],'p',1),[],[]);
  
  if mapping_task(argin,'definition')
    
    W = define_mapping(argin,'untrained',mapname);
    
	elseif mapping_task(argin,'training')			% Train a mapping.

    [a,kernel,nu,Options] = check_for_old_call(argin);

    DefOptions.mean_centering       = 1;
    DefOptions.pd_check             = 1;
    DefOptions.bias_in_admreg       = 1;
    DefOptions.allow_ub_bias_admreg = 1;
    DefOptions.pf_on_failure        = 1;
    DefOptions.multiclass_mode      = 'single';    

    Options = updstruct(DefOptions,Options,1);
		
  	islabtype(a,'crisp');
		a = testdatasize(a,'objects'); 
    
    % remove too small classes, escape in case no two classes are left
    [a,m,k,c,lablist,L,W] = cleandset(a,1); 
    if ~isempty(W), [J,C,nu,alginf] = deal(NaN,NaN,NaN,NaN); return; end
	
		% The SVC is basically a 2-class classifier. More classes are
  	% handled by mclassc.
    if c == 2   % two-class classifier   
      
      if isempty(nu)
        nu = 2*min(max(testk(a,1),0.01),(0.8*min(classsizes(a))/size(a,1)));
      end
      nlab = getnlab(a);

			if (isnan(nu))                      % optimise trade-off parameter
				defs = {proxm([],'p',1),[],[]};
				parmin_max = [0,0;0.001,0.999;0,0]; % kernel and Options can not be optimised
				[W,J,nu,C,alginf] = regoptc(a,mfilename,{kernel,nu,Options},defs,[2],parmin_max,testc([],'soft'));
			else
      	% Compute the parameters for the optimisation:
      	y = 3 - 2*nlab;
        if isequal(kernel,0)
          s = [];
          u = [];
          in_size = 0; % to allow old and new style calls
        else
          if Options.mean_centering
            u = mean(a);         % shift origin for better accuracy
            a = a - repmat(u,[m,1]);
          else
            u = [];
          end
          in_size = k;
        end
        K = compute_kernel(a,a,kernel);
        K = min(K,K');   % make sure kernel is symmetric
        [v,J,nu,C] = nusvo(+K,y,nu,Options);
        
        % Store the results:
        if ~isequal(kernel,0)
          s = a(J,:);
        end   
       
        % Store the results, use SVC for execution
      	W = prmapping('svc','trained',{u,s,v,kernel,J},getlablist(a),in_size,2);
      	W = cnormc(W,a);
      	W = setcost(W,a);
      	alginf.svc_type = 'nu-SVM';
      	alginf.kernel = kernel;
      	alginf.C   = C;
      	alginf.nu  = nu;
      	alginf.nSV = length(J);
      	alginf.classsizes = [nnz(y==1), nnz(y==-1)];
      	alginf.pf = isnan(C);
			end
  		
  	else   % multi-class classifier:
	    [W,J,nu,C,alginf] = mclassc(a,prmapping(mfilename,{kernel,nu,Options}),'single');   
      W = allclass(W,lablist,L);     % complete classifier with missing classes
		end
		
 		W = setname(W,mapname);

	end
	
  return;
  
function K = compute_kernel(a,s,kernel)

	% compute a kernel matrix for the objects a w.r.t. the support objects s
	% given a kernel description

	if  isstr(kernel) % routine supplied to compute kernel
		K = feval(kernel,a,s);
	elseif iscell(kernel)
		K = feval(kernel{1},a,s,kernel{2:end});
	elseif ismapping(kernel)
		K = a*prmap(s,kernel);
	elseif kernel == 0 % we have already a kernel
		K = a;
	else
		error('Do not know how to compute kernel matrix')
	end
		
	K = +K;
		
return

function [a,kernel,NU,par] = check_for_old_call(argin)

[a,kernel,NU,par] = deal(argin{:});
if ischar(kernel) && exist(kernel,'file') ~= 2
  kernel = proxm(kernel,NU);
  NU = par;
end

