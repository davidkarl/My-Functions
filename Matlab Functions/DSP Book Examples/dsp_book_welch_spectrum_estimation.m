T=1000;
blocana=64;
Lfft=1024;
freq=(0:Lfft-1)/Lfft;

%signal generation:
w=randn(1,T);
x=filter([1,0.5,0.02,0.01],1,w);
xfth=20*log10(abs(fft([1;0.5;0.02;0.01],Lfft)));
[xfw,gamma]=welch(x,blocana,'ham',Lfft,0.95);
[xfp,freqp]=smperio(x,16,'t');
xfw=10*log10(xfw);
xfp=10*log10(xfp);
df1=-10*log10(1+gamma);
df2=-10*log10(1-gamma);
plot(freq,xfw,freq,xfw+df1,'g-.',freq,xfw+df2,'g-.',freq,xfth);
set(gca,'xlim',[0,0.5]);
hold on;
plot(freqp,xfp,'r');
legend('welch','welch upper','welch lower','theoretical','smoothed psd');
hold off














