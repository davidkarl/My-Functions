function [a]=estARrls(x,P)
%dsp book estimate ARp using RLS algorithm
%x=signal
%P=model order
%a=[1,a1,...ap]
N=length(x);
x=x(:);
x=x-ones(1,N)*x/N;
hn=zeros(P,1);
delta=10^-6;
Qn=eye(P)/delta;
%correlations method:
x=[zeros(P,1);x;zeros(P,1)];
for k=P+1:N+2*P
   xn=x(k-1:-1:k-p);
   yn=x(k);
   Gn=Qn/(1+xn'*Qn*xn);
   Kn=Gn*xn;
   en=yn-xn'*hn;
   hn=hn+Kn*en;
   Qn=Qn-Kn*xn'*Qn;
end
a=[1;-hn];







