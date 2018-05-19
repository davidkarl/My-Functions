%dsp book finding two harmonics using lsq
T=10;
f01=0.12;
f02=0.61;
tps=[0:T-1]';

%signal:
s=1.5*exp(2*pi*1i*f01*tps)+exp(2*pi*1i*f02*tps);
SNR=15;
sigma2=(s'*s/T)/(10^(SNR/10));

%noised signal:
xb=s+sqrt(sigma2)*randn(T,1);
Lf=70;
f1=(0:Lf-1)/Lf;
f2=f1;

% mm=exp(2*pi*1i*f1*tps);
[X,Y]=meshgrid(f1,tps); 
mm=exp(2*pi*1i*(X.*Y));
yy=zeros(Lf,Lf);
 
for k1=1:Lf
    for k2=1:k1-1
       E=[mm(:,k1),mm(:,k2)];
       yy(k1,k2)=abs(xb'*E*pinv(E)*xb);
    end 
end

subplot(1,2,1); mesh(f1,f2,yy); view([115,35]);
subplot(1,2,2); plot(f1,abs(fft(xb,Lf))); grid



