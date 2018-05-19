%PKSTATSSVC Automatic radial basis STATSSVM optimising the Parzen kernel
%
%   [W,KERNEL] = PKSTATSSVC(A,ALF)
%   [W,KERNEL] = A*PKSTATSSVC([],ALF)
%   [W,KERNEL] = A*PKSTATSSVC(ALF)
%
% INPUT
%   A   Dataset
%   ALF Parameter, default 0.5
%
% OUTPUT
%   W       Mapping: Radial Basis Support Vector Classifier
%   KERNEL  Untrained mapping, representing the optimised kernel
%
% DESCRIPTION
% This routine provides a radial basis support vector classifier based on
% STATSSVC. It uses a radial basis kernel supplied by PROXM with a kernel
% width SIGMA found by PARZENC. The kernel width used is ALF*3*SQRT(2)*SIGMA. 
% This is much faster than the gridsearch used by RBSTATSSVC and performs
% about equally well.
%
% Note that STATSSVC is basically a two-class classifier and solves
% multi-class problems by MCLASSC, generating a set of one-against-rest
% classifiers. PKSTATSSVC, however, supplies a single kernel. For some
% problems a proper prescaling of the data, e.g. by SCALEM, amy thereby be
% appropriate.
%
% SEE ALSO (<a href="http://37steps.com/prtools">PRTools Guide</a>)
% DATASETS, MAPPINGS, STATSSVC, RBSTATSSVC, PARZENC, SVC, MCLASSC, SCALEM

% Copyright: R.P.W. Duin, r.p.w.duin@37steps.com

function [w,kernel] = pkstatssvc(varargin)

  argin = shiftargin(varargin,'scalar');
  argin = setdefaults(argin,[],0.5);
  mapname = 'StatsPKSVM';
  if mapping_task(argin,'definition')
    w = define_mapping(argin,'untrained',mapname);
    
  elseif mapping_task(argin,'training')			% Train a mapping.
  
    [a,alf] = deal(argin{:});

    if size(a,1) <= 1000
      [v,sig] = parzenc(a);
    else
      [v,sig] = parzenc(gendat(a,1000));
    end
    kernel = proxm([],'r',alf*3*sig*sqrt(2));
    w = statssvc(a,kernel);
    w = setname(w,mapname);
  end

