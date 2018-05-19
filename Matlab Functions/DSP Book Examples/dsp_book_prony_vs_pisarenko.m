%dsp book prony vs. pisarenko:
N=25;
Am=[2,1.5,1];
F=[0.2,0.225,0.3];
P=length(Am);
s=Am*cos(2*pi*F'*(0:N-1));
SNR=40;
sigma2=(s*s'/N)/(10^(SNR/10));
x=s+sqrt(sigma2)*randn(1,N);
D=toepl(x(2*P+1:N),x(2*P+1:-1:1));
R=D*D';

%prony:
U=[1;zeros(2*P,1)];
B=inv(R)*U;
lambda=U'*B;
A=B/lambda;
fkProny=angle(rotts(A))/(2*pi);
FestProny=sort(fkProny(find(fkProny>0)));

%pisarenko:
[Vvect,Val]=eig(R);
[ordVal,iV]=min(diag(Val));
Vmin=Vvect(:,iV);
fkPisar=angle(roots(Vmin))/(2*pi);
FestPisar=sort(fkPisar(find(fkPisar>0)))';
F,FestProny,FestPisar






