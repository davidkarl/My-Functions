%dsp book equalization using a P-coefficient FIR filter
N=1000;
x0=randn(1,N);
x1=0.2*randn(1,N);
x=[x0';x1';x0'];
subplot(4,1,1); plot(x); grid

%filtering and addition of noise
h0=[1,0.7]; v0=filter(h0,1,x0);
h1=[1,0.3]; v1=filter(h1,1,x1);
v=[v0';v1';v0']; N=length(v);
SNR=30;
Px=x'*x/N;
sigma=sqrt(Px)*10^(-SNR/20);
b=sigma*randn(N,1);
y=v+b;
subplot(4,1,2); plot(y); grid;

%equalization using a p coefficient FIR filter:
P=6;
HestS=zeros(P,1); %standard LMS
HestN=zeros(P,1); %normalized LMS
enS=zeros(N-P+1,1); %standard LMS error
enN=zeros(N-P+1,1); %normalized LMS error

%implementing the LMSs
muS=0.06;
muN=0.08;
%forget factor:
alpha=0.05;
umalpha=1-alpha;
%Py(n) initial:
pyn=y(1:P-1)'*y(1:P-1)/P;

%Algorithms:
for n=P:N
    %standard:
    en0S=x(n)-HestS'*y(n:-1:n-P+1);
    HestS=HestS+muS*en0S*y(n:-1:n-P+1);
    enS(n-P+1)=en0S;
    %normalized:
    en0N=x(n)-HestN'*y(n:-1:n-P+1);
    %two expressions for the estimation of Py:
    pyn=y(n:-1:n-P+1)'*y(n:-1:n-P+1)/P;
    %pyn=umalpha*pyn+alpha*y(n)*y(n);
    HestN=HestN+muN*en0N*y(n:-1:n-P+1)/pyn;
    enB(n-P+1)=en0N;
end

%displaying results:
en2N=enN.^2;
en2S=enS.^2;
moy=100;
hmoy=ones(1,moy)/moy;
en2moyS=filter(hmoy,1,en2S(1:N-P+1));
en2moyN=filter(hmoy,1,en2N(1:N-P+1));
endBS=10*log10(en2moyS(moy:N-P+1));
endBN=10*log10(en2moyN(moy:N-P+1));
subplot(2,1,2); plot(endBS); grid
hold on;
plot(endBN,'r'); hold off;













