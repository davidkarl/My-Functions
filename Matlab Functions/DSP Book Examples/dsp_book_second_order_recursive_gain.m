%dsp book second order recursive filter gain

%gain as a function of the phase with a constant modulus:
Lfft=1024;
freq=(0:Lfft-1)/Lfft;
modp=0.9;
theta=(20:10:80);
theta=theta*pi/180;
nbph=length(theta);
a1=-2*modp*cos(theta);
a2=modp^2*ones(1,nbph);
AA=[ones(1,nbph);a1;a2];
Df=fft(AA,Lfft);
Hf=-20*log10(abs(Df));
plot(freq(1:Lfft/2),Hf(1:Lfft/2,:));
grid; 

%gain as a function of the modulus:
Lfft=1024;
freq=(0:Lfft-1)/Lfft;
modp=[0.1:0.2:0.9,0.95,0.98]; 
theta=30*pi/180;
nbph=length(modp);
a1=-2*modp*cos(theta);
a2=modp.^2;
AA=[ones(1,nbph);a1;a2];
Df=fft(AA,Lfft);
Hf=-20*log10(abs(Df));
plot(freq(1:Lfft/2),Hf(1:Lfft/2,:));
grid





