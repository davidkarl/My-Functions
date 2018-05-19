%dsp book prony method for harmonic estimation
Am=[2,1.5,1];
P=length(Am);
F=[0.2,0.225,0.3];
N=25;
s=Am*cos(2*pi*F'*(0:N-1));
SNR=40;
sigma2=(s*s'/N)/(10^(SNR/10));

%noisy signal:
x=s+sqrt(sigma2)*randn(1,N);
D=toeplitz(x(2*P+1:N),x(2*P+1:-1:1));
R=D'*D;
U=[1;zeros(2*P,1)];
B=inv(R)*U;
lambda=U'*B;
A=B/lambda;

%verification:
Lfft=256;
fq=(0:Lfft-1)/Lfft;
gf=1./abs(fft(A,Lfft));
subplot(2,1,1); plot(fq(1:Lfft/2),gf(1:Lfft/2)); grid
xf=abs(fft(x,Lfft));
subplot(2,1,2); plot(fq(1:Lfft/2),xf(1:Lfft/2)); grid



