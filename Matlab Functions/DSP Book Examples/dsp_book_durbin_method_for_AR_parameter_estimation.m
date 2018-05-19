sigma2=9;
w=sqrt(sigma2)*randn(10000,1);
hh=[1;0.3]; %minimuim phase case
%hh=[0.3;1]; %non-minimum phase case
Q=length(hh)-1;
x=filter(hh,1,w);
[b,s2]=durbin(x,Q); 
[hh,b],[sigma2, s2]










