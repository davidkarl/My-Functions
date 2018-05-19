function [xhat,XHAT,ALPHA]=kallman(U,d,varargin)
%
%[xhat,XHAT,ALPHA]=kallman(x,d,xhat0,lamb,K0,sig)

%[xhat,XHAT,ALPHA]=kallman(U,d,xhat0,lamb,K0,sig)
%
%Implements unforced kallman filter. Tap weights are the state vectors
%inputs are C(t). Parameters are:
% 
% U       - (NxM) Signals that will be used to predict d.
% d       - (Nx1) Desired response
% xhat0   - (Mx1) Initial Weights (optional)
% lamb    - (1x1) Forgetting Factor (optional)
% K0      - (Mx1) Initialization matrix (optional)
% sig     - (NxM) Noise estimates of each U channel (optional)
% 
% 
% xhat    - (1xM) Final estimated weights
% XHAT    - (NxM) Weight learning curve
% ALPHA   - (Nx1) Error curve



%Implements unforced kallman filter. Tap weights are the state vectors
%inputs are C(t)
% Copyright (C) 2010 Ikaro Silva
% 
% This library is free software; you can redistribute it and/or modify it under
% the terms of the GNU Library General Public License as published by the Free
% Software Foundation; either version 2 of the License, or (at your option) any
% later version.
% 
% This library is distributed in the hope that it will be useful, but WITHOUT ANY
% WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
% PARTICULAR PURPOSE.  See the GNU Library General Public License for more
% details.
% 
% You should have received a copy of the GNU Library General Public License along
% with this library; if not, write to the Free Software Foundation, Inc., 59
% Temple Place - Suite 330, Boston, MA 02111-1307, USA.
% 
% You may contact the author by e-mail (ikaro@ieee.org).

%Initialize all parameters
[N,M]=size(U);
lamb=1;

if (nargin>2 && ~isempty(varargin{1}) )
    xhat=varargin{1};
    xhat=xhat(:);
else
    xhat=rand(M,1)*0.0001;
end
if (nargin>3 && ~isempty(varargin{2}))
    lamb=varargin{2};
end
if (nargin>4 && ~isempty(varargin{3}))
    K=varargin{3};
else
    K=corrmtx(xhat,M-1);
    K=K'*K;
end
if (nargin>5 && ~isempty(varargin{4}))
    sig=varargin{4};
else
    sig=ones(N,M);  %measureement noise (assumed diagonal)!!!!
end


%Do the training
lamb=lamb^(-0.5);
ALPHA=zeros(N,1);
XHAT=zeros(N,M);

for n=1:N
    
    u=U(n,:)';   
    den=diag(1./(u'*K*u + sig(n,:)));
    g= lamb*den*K*u ;
    alpha = d(n) - u'*xhat;
    xhat= lamb *xhat + g*alpha;
    K = (lamb^2)*K - lamb*g*u'*K;

    ALPHA(n)=alpha;
    XHAT(n,:)=xhat';
    
end
