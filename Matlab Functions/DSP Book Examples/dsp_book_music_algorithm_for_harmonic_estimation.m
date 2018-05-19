%SHOW musicFFT:
nfft=1024;
freq=(0:nfft-1)/nfft;
SNRdB=50;
N=80;
tps=(0:N-1); %N=number of samples
fq=[0.2,0.21];
p=length(fq);
alpha=[1,0.2];
%signal:
sig=alpha*cos(2*pi*fq'*tps);
vseff=std(sig);
siggmab=sqrt(10^(-SNRdB/10))*vseff/sqrt(2);
b=siggmab*randn(1,N);
x=sig+b; %noised signal
%use musicFFT:
[racm,QQfft]=musicFFT(x,2*p);
fqm=angle(racm)/(2*pi);
fqaux=sort(fqm); %sort the frequencies
fqmo=fqaux(p+1:2*p);
regang=2*pi*fqmo*tps;
RR=[cos(regang)', sin(regang)'];
ab=RR\x';
alphamo=sqrt(ab(1:p).^2+ab(p+1:2*p).^2);
%display the DTFT:
subplot(2,1,1);
plot(freq,20*log10(abs(fft(x,nfft))/N));
set(gca,'xlim',[0,0.5]);
zoom xon;
grid;
LQ=length(QQfft);
subplot(2,1,2);
plot([0:LQ-1]/LQ,-20*log10(abs(QQfft)));
set(gca,'xlim',[0,1/2]); grid;
%display the results
disp(sprintf('SNR:\t %5.4g dB',SNRdB));
disp(sprintf('Number of samples: \t %3i',N));
disp(sprintf('Number of sines: \t %3i',p));
disp('True values:');
disp(sprintf('Freq. = %5.4g\t ampl. =%5.4g\t \n',[fq;alpha]));
disp('estimated values:');
disp(sprintf('\t freq. = %5.4g\t ampl. = %5.4g\t \n',[fqmo';alphamo']));
 


%SHOW music:
N=80;
tps=(0:N-1);
fq=[0.2,0.21];
alpha=[1,0.2]; 
p=length(fq);
%signal
sig=alpha*cos(2*pi*fq'*tps);
vseff=std(sig)/sqrt(2);
%SNR:
SNR=(10:2:20);
lSNR=length(SNR);
eqmfq=zeros(lSNR,p);
L=30; %number of trials
for ii=1:lSNR
   SNRdB=SNR(ii);
   siggmab=sqrt(10^(-SNRdB/10))*vseff;
   fqmo=zeros(p,L);
   alphamo=zeros(p,L);
   for jj=1:L
      b=siggmab*randn(1,N);
      %noised signal:
      x=sig+b;
      racm=music(x,2*p);
      fqm=angle(racm)/(2*pi);
      fqaux=sort(fqm);
      fqmo(:,jj)=fqaux(p+1:2*p);
   end
   dfqmo=fqmo-fq'*ones(1,L);
   eqmfq(ii,:)=std(dfqmo');
end
%displaying results:
subplot(1,2,1);
plot(SNR,eqmfq(:,1),'o'); 
grid
hold on;
plot(SNR,eqmfq(:,1)); 
hold off;
subplot(1,2,2);
plot(SNR,eqmfq(:,2),'o');
grid
hold on;
plot(SNR,eqmfq(:,2));
hold off;










