%dsp book channel identification
h=[1,0.6,0.3]'; P=length(h); %theoretical channel
N=4000; %number of steps

%signal generation:
x=randn(N,1);
v=filter(h,1,x);
Pv=v'*v/N;
SNR=20;
b=sqrt(Pv*10^(-SNR/10))*randn(N,1);
y=v+b; %noisy observation:

%LMS algorithm:
mu=0.002;
hest=zeros(P,1);
en=zeros(N-P+1,1);

for n=P:N
   en0=y(n)-hest'*x(n:-1:n-P+1);
   hest=hest+mu*en0*x(n:-1:n-P+1);
   en(n-P+1)=en0;
end

%smoothing of the error over 200 points:
en2=en.^2;
moy=200;
hmoy=ones(1,moy)/moy;
en2moy=filter(hmoy,1,en2(1:N-P+1));
endb=10*log10(en2moy(moy:N-P+1));
plot(endb);
grid
[h,hest]




