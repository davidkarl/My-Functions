%assuming frequency is known and what i want to do is find the jammer's
%amplitdue and phase

sn = wavread('rodena blabla.wav');
sn=sn(:,1);
N=length(sn);

%modulation (A=1 for a pure jammer)
Fb=1000;
Fe=8000;
fb=Fb/Fe;
Fm=20;
fm=Fm/Fe;
A=1;
k=0.005;
b=A*(1+k*cos(2*pi*fm*(1:N)')) .* cos(2*pi*fb*(1:N)');

%jammed signal
x=sn+b;
%lms implementation:
mu=0.01;
P=30;
gch=zeros(2*P,1);
%schap is the reconstructed signal:
tic
schap=zeros(N,1);
for n=P:N
   gY=[cos(2*pi*fb*(n:-1:n-P+1)') ; sin(2*pi*fb*(n:-1:n-P+1)')];
   en0=x(n)-gch'*gY;
   gch=gch+mu*en0*gY;
   schap(n)=en0; 
end
toc
subplot(3,1,1); plot(sn); grid; subplot(3,1,2); plot(x); grid
subplot(3,1,3); plot(schap); set(gca,'ylim',[-1,1]); grid









