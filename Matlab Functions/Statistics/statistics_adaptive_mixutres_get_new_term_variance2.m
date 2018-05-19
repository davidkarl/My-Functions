function newvar = statistics_adaptive_mixutres_get_new_term_variance2(mus,pies,vars,x,nterms)
% SETVARM   Helper function for adapative mixtures.

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox


%  function  newvar =  setvarm(mus,pies,vars,x,nterms)
% this function will update the variances
% in the recursive amde
% call with nterms-1, since new term is based only on
% previous terms stuff



totprob=0;
sievebd=1000;		% bounding the parameter space

posterior=zeros(1,nterms);

[d,c]=size(mus);
newvar=zeros(d,d);

for i=1:nterms	%loop to find total prob in denominator (hand, pg 37)
  posterior(i)=pies(i)*csevalnorm(x,mus(:,i)',vars(:,(i-1)*d+1:i*d));
  totprob=totprob+posterior(i);
end


posterior=posterior/totprob;

for i=1:nterms
  newvar=newvar+posterior(i)*vars(:,(i-1)*d+1:i*d);
  if det(vars(:,(i-1)*d+1:i*d))<1/(sievebd*nterms);
  	vars(:,(i-1)*d+1:i*d)=eye(d,d)*sqrt(1/(sievebd*nterms));
  end


end



