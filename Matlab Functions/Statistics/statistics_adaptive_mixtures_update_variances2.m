function vars = statistics_adaptive_mixtures_update_variances2(x,pies,mus,vars,posterior,nterms,n)
% RVARUPM   Helper function for adapative mixtures.

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox

inertvar=10;
betan=1/(n);
sievebd=1000;		% bounding the parameter space

[d,c]=size(mus);

for i=1:nterms
  denom=(1/betan)*pies(i)+inertvar;
  vars(:,(i-1)*d+1:i*d)=vars(:,(i-1)*d+1:i*d)+posterior(i)*((x'-mus(:,i))*(x'-mus(:,i))'...
  	-vars(:,(i-1)*d+1:i*d))/denom;
  if det(vars(:,(i-1)*d+1:i*d))<1/(sievebd*nterms);
  	vars(:,(i-1)*d+1:i*d)=eye(d,d)*sqrt(1/(sievebd*nterms));
  end

end
