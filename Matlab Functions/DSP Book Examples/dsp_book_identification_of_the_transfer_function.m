Lfft=1024;
freq=(0:Lfft-1)/Lfft;
num=10;
den=2;
F=(0:0.01:0.5)';
Lf=length(F);

%Passband:
fcP=0.2;
LfcP=ceil(2*fcP*Lf);
%ideal lowpass:
H=[ones(LfcP,1);zeros(Lf-LfcP,1)];
%derivative:
%H=1i*pi*F;
%weighting:
%deltaP=0.03; deltaA=0.1;
%weighting coeffts:
%usP=1/deltaP; usA=1/deltaA;
%stopband:
%fcA=0.21; LfcA=ceil(2*fcA*LF);
%W=[usP*ones(1,LfcP), zeros(1,LfcA-LfcP), usA*ones(1,LF-LfcA)];
 
[b1,a1]=hftoz(H,F,num,den,'real');
hf1=abs(fft(b1,Lfft))./abs(fft(a1,Lfft));
hf1=20*log10(hf1/hf1(1));
plot(freq,hf1,'-',F,20*log10(H));
grid;
axis([0,0.5,-40,10]);











