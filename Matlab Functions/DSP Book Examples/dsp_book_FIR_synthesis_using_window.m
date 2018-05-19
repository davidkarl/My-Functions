%bandpass filter:
Lfft=1024;
fq=(0:Lfft-1)/Lfft;
f0=0.2;
fb=0.1;
N=11;
P=fix(N/2);
R=rem(N,2);
h=rif(N,fb/2);
if R==0,
    D=(-P:P-1)+1/2;
else
    D=(-P:P);
end
g=2*h.*cos(2*pi*f0*D); 
gf=fft(g,Lfft); 
agf=abs(gf);
phigf=angle(gf);
figure(1);
plot(fq(1:Lfft/2),agf(1:Lfft/2)); grid
figure(2);
plot(fq(1:Lfft/2),phigf(1:Lfft/2)); grid






