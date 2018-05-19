function [epsF,epsB] = lattice_analysis(xn,ki)
%dsp book lattice-analysis
%xn=signal
%ki=reflection coefficients (k1,...kP)
%epsF=forward error
%epsB=backward error

N=length(xn);
epsF=zeros(N,1);
epsB=zeros(N,1);
P=length(ki);
eB=zeros(P,1);
eBm1=zeros(P,1);

for nn=1:N
   eF=xn(nn);
   for pp=1:P
      eFp=eF+ki(pp)*eBm1(pp);
      eB(pp)=ki(pp)*eF+eBm1(pp);
      eF=eFp;
   end
   eBm1=[xn(nn);eB(1:P-1)];
   epsF(nn)=eFp;
   epsB(nn)=eB(P);
end





