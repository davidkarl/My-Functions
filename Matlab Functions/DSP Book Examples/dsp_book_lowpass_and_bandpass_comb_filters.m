%dsp book lowpass and bandpass comb filters:
M=15;
m=(0:M-1);
Lfft=512;
fq=(0:Lfft-1)/Lfft-1/2;
fq1=(Lfft/2+1:Lfft);
fq2=(1:Lfft/2);

ht=ones(1,M);
k=4; 
gt=2*ht.*cos(2*pi*k*m/M);
hf=abs(fft(ht,Lfft)); hf=10*log10(hf/max(abs(hf)));
gf=abs(fft(gt,Lfft)); gf=10*log10(gf/max(abs(gf)));
subplot(2,1,1); plot(fq,[hf(fq1),hf(fq2)]);
set(gca,'ylim',[-20,0]); grid
subplot(2,1,2); plot(fq,[gf(fq1),gf(fq2)]);
set(gca,'ylim',[-20,0]); grid





