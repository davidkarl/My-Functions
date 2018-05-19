function [realt,xout]=Crepind(A,bc,Ts,tmax,x0)
%dsp book step response of a linear system
% (A,b,c)=state representation
%Ts=sampling frequency
% tmax=observation duration
%x0=initial state
%realt=real time
%xout=response
npts=floor(tmax/Ts);
[N,N]=size(A); %system order
%sampling frequency =1/Ts
Ae=[A,b;zeros(1,N+1)]*Ts;
Aexp=expm(Ae);
phi=Aexp(1:N,1:N);
psib=Aexp(1:N,N+1);
tps=[0:npts-1];
realt=tps*Ts;
xout=zeros(1,npts);
xx=x0; %initial conditions
xout(1)=c*xx;
for k=2:npts
   xx=phi*xx+psib;
   xout(k)=c*xx;
end
return

%DOESN'T INTEREST ME FOR NOW THIS IS FOR CONTINUOUS TIME SYSTEMS DESCIRBED
%BY SYSTEM MATRIX A






