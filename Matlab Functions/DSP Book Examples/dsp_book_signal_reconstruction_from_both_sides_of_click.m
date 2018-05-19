%dsp book signal reconstruction from both sides of click
pos=input('click position:');
lsig=length(sig);
tps=[0:lsig-1];
sig=reshape(sig,lsig,1);
m=15;
ell=pos-7;
X0=sig(ell-K:ell-1);
X1=sig(ell+m:ell+m+K-1);
colT=[aest(K);zeros(m+K-1,1)];
ligT=[aest(K:-1:1)', zeros(1,m+K)];
T=toplitz(colT,ligT);
A0=T(:,1:K);
B=T(:,K+1:K+m);
A1=T(:,K+m+1:2*K+m);
X=A0*X0+A1*X1;

%solve the system:
Y=-B\X;
sigr=sig;
sigr(ell:ell+m-1)=Y;
plot(tps,sig,'-r',tps,s,'b',tps,sigr,':y');
grid;






