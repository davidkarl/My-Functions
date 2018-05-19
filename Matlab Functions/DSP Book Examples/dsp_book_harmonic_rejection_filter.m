%dsp book harmonic component rejection filter:
%the crux of this method is to place a zeros pair on the unit circle (rho=1) and
%to put a poles pair very close to the same point on the unit circle such
%that there will indeed be a rejection of the wanted frequency (using the
%zero) but the gains of all the other frequencies will be approximately one

nfft=256;
freq=[0:nfft-1]/nfft;
phi=pi/4;
ro=0.9;
num=[1,-2*cos(phi),1];
den=[1,-2*ro*cos(phi),ro*ro];
k=sum(den)/sum(num); num=k*num; %normalization
snum=fft(num,nfft);
sden=fft(den,nfft);
spec=snum./sden;
plot(freq,abs(spec));
grid






