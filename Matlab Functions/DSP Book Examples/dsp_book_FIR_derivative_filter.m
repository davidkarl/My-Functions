N=12;
Fs=4000;
nfft=512;
freq=(0:nfft-1)/nfft*Fs;
F0=300;
T=100;
t=(0:T-1)/Fs;

%original:
x=sin(2*pi*F0*t);
subplot(3,2,1); 
plot(t,x); 
grid
axis([0,(T-1)/Fs,-1.2,1.2]); 
title('x(t)');
%theretical derivative:
xp=2*pi*F0*cos(2*pi*F0*t); %result
ordm=1.2*2*pi*F0;
subplot(3,2,2); 
plot(t,xp); 
grid
axis([0,(T-1)/Fs,-ordm,ordm]); 
title('x''(t)');
%digital derivative:
[y,hder]=deriv(N,x);
y=Fs*y;
subplot(3,2,3); 
plot(t,y); 
grid
axis([0,(T-1)/Fs,-ordm,ordm]); 
title('y(t)');
%delay due to the filter:
subplot(3,2,4); 
plot(t,xp,'b',t-N/Fs,y); 
grid;
axis([0,(T-1)/Fs,-ordm,ordm]); 
title('y(t-N/Fs)');
%gain of the derivative filter:
hders=fft(Fs*hder,nfft);
subplot(3,1,3)
plot(freq,abs(hders),[0,Fs/2],[0,Fs*pi]);
axtemp=axis;
axis([0,Fs/2,axtemp(3:4)]);
grid










