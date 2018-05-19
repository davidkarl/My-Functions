%dsp book denoising an AR1 signal using kalman
N=200;
mtime=(0:N-1);
xch=zeros(N,1);
a=0.9; %time constant for the AR1
sigmab=1; %modelling noise with variance 1

%trajectory:
b=sigmab*randn(N,1);
x=filter(1,[1,-a],b);
sigmau=3; %observation noise: variance=9
y=x+sigmau*randn(N,1); %observation
saux=9*sigmau^2;

%tracking: 
a2=a*a;
rho=(sigmab/sigmau)^2;
G(1)=rho/(1-a2);

for nn=2:N
    G(nn)=(rho+a2*G(nn-1))/(1+rho+a2*G(nn-1));
    xch(nn)=a*xch(nn-1)+G(nn-1)*(y(nn)-a*xch(nn-1));
end
plot(mtime,x,'-',mtime,y,':',mtime,xch,'o');



