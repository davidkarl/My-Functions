%PKSVC Automatic radial basis SVM, using NUSVC and the Parzen kernel
%
%   [W,KERNEL,NU,C] = PKSVC(A,ALF)
%   [W,KERNEL,NU,C] = A*PKSVC([],ALF)
%   [W,KERNEL,NU,C] = A*PKSVC(ALF)
%
% INPUT
%   A   Dataset
%   ALF Parameter, default 1
%
% OUTPUT
%   W       Mapping: Radial Basis Support Vector Classifier
%   KERNEL  Untrained mapping, representing the optimised kernel
%   NU      Resulting value for NU from NUSVC (W = NUSVC(A,KERNEL,NU))
%   C       Resulting value for C (W = SVC(A,KERNEL,C)
%
% DESCRIPTION
% This routine provides a radial basis support vector classifier based on
% NUSVC (which estimates NU using the leave-one-out 1NN error) and
% estimates the kernel width SIGMA by the the value found by PARZENC. The
% kernel width used is ALF*3*SQRT(2)*SIGMA. This is much faster than the
% gridsearch used by RBSVC and performs about equally well.
%
% SEE ALSO (<a href="http://37steps.com/prtools">PRTools Guide</a>)
% DATASETS, MAPPINGS, SVC, NUSVC, RBSVC, PARZENC, LIBSVC

% Copyright: R.P.W. Duin, r.p.w.duin@37steps.com

function [w,kernel,nu,C] = pksvc(varargin)

  argin = shiftargin(varargin,'scalar');
  argin = setdefaults(argin,[],1);
  mapname = 'PK-SVC';
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
    [w,J,nu,C] = nusvc(a,kernel);
    w = setname(w,mapname);
    
  end

