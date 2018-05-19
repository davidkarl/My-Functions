function [xn,epsB]=lattice_synthesis(epsF,ki)
%dsp book lattice-synthesis:
%epsF=forward error
%ki=reflection coefficients (k1,...kP)
%xn=reconstructed signal
%epsB=backward error
N=length(epsF);
xn=zeros(N,1);
epsB=zeros(N,1);
P=length(ki);
eB=zeros(P,1);
eBm1=zeros(P,1);

for nn=1:N
   eF=epsF(nn);
   for pp=P:-1:1
      eF=eF-ki(pp)*eBm1(pp);
      eB(pp)=eBm1(pp)+ki(pp)*eF;
   end
   xn(nn)=eF;
   eBm1=[eF;eB(1:P-1)];
   epsB(nn)=eB(P);
end




