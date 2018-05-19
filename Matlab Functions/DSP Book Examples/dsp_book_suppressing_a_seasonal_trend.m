N=100;
B=randn(N,1);
a=4;
f0=0.01;
phi=pi/6;
tseason=3*cos(2*pi*f0*[0:N-1]'-phi);
X=a+tseason+B;
Res=trendseason(X,f0);
subplot(3,1,1); plot(B); grid; set(gca,'ylim',[-4,4]);
subplot(3,1,2); plot(X); grid;
subplot(3,1,3); plot(Res); grid; set(gca,'ylim',[-4,4]);


