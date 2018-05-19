%dsp book estimate impulse response using RLS
SNR=10;
h0=[1;-5/2;1];
P=length(h0);
N=200;
x=randn(N,1);
sigmab=10^(-SNR/20);

yssb=filter(h0,1,x); %output
y=yssb+sigmab*randn(N,1); %noised output

%RLS algorithm:
Qn=10^7*eye(P); %initialization
hn=zeros(P,1);
xnp1=zeros(P,1);
for k=P:N
   xnp1(:)=x(k:-1:k0P+1);
   Qnh=Qn*xnp1;
   cc=1+xnp1'*Qnh;
   Kn=Qnh/cc;
   en(k)=(y(k)-xnp1'*hn);
   hn=hn+Kn*en(k);
   dh(k-P+1)=(hn-h0)'*(hn-h0);
   Qn=Qn-(Qnh*xnp1'*Qn/cc);
end
plot(10*log10(dh));
grid

%theoretical limi:
limT=10*log10(P/N)-SNR;
hold on;
plot([0,N-P+1],[limT,limT]);
hold off;








