%dsp book pisarenko method:
Lfft=256;
freq=(0:Lfft-1)/Lfft;
N=25;
Am=[2,1.5,1];
F=[0.2,0.225,03];
P=length(Am);

%original signal:
s=Am*cos(2*pi*F'*(0:N-1));
SNR=40;
sigma2=(s*s'/N)/(10^(SNR/10));

%noisy signal:
x=s+sqrt(sigma2)*randn(1,N);
D=toepl(x(2*P+1:N),x(2*P+1:-1:1)); 
R=D*D';
[Vvect,Val]=eig(R);
[odVal,iV]=min(diag(Val));
Vvect=Vvect(:,iV);
vf=1./abs(fft(Vvect(:,1),Lfft));
plot(freq(1:Lfft/2),vf(1:Lfft/2));
grid;






