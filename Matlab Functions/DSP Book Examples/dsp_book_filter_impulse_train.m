%dsp book filter impulse train:
Fs=8000;
F0=120;
M=round(Fs/F0); %period in sample number

%for ideal pulse P=1
%in all cases P must be much smaller than M
P=1;
pulse=ones(P,1);
%another pulse shape: [0.2;0.7;1;0.3;0.1];
nbpulses=20;
lx=nbpulses*M;
xs=zeros(lx,1);
for tt=0:nbpulses-1
   ii=tt*M+1;
   iifin=ii+P-1;
   xs(ii:iifin)=pulse;
end
K=10; %try any shift
xs=[zeros(K,1);xs]; lx=length(xs);
xs=xs*sqrt(lx)/sum(xs);
Px=xs'*xs/lx;

subplot(4,1,1);
plot((1:lx)/Fs,xs);
nfft=2^nextpow2(lx);
freq=Fs*(0:nfft-1)/nfft;
Xf=abs(fft(xs,nfft))/sqrt(lx);
subplot(4,1,1); plot(freq,Xf); set(gca,'xlim',[0,Fs/2]);

%filtering:
aa=[1;-1.6;0.9];
Hf=1./abs(fft(aa,nfft));
ys=filter(1,aa,xs);
Yf=abs(fft(ys,nfft))/sqrt(lx);

subplot(4,1,3); plot(freq,Yf);
hold on; plot(freq,Hf,'r'); hold off;
set(gca,'xlim',[0,Fs/2]);
[aae,sse]=xtoa(ys,2);
[aa,aae],[Px,sse]
subplot(4,1,4); 
hatxs=filter(aae,1,ys);
plot((1:lx)/Fs,hatxs);




