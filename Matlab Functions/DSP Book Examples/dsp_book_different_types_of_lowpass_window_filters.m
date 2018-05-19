%dsp book different types of lowpass window filters
FFT_size = 256;
frequency_vec =[0:FFT_size-1]'/FFT_size;
Fc=1/8;
Nt=56;

%Type 1 (hn odd):
Mt=Nt+1;
M=floor(Mt/2);
n=[-M:M]'; 
hnI=sin(2*pi*n*Fc)./(pi*n);
hnI(M+1)=2*Fc;
hnIs = fft(hnI,FFT_size);
hrIs=abs(hnIs);

%Type 2 (hn even):
Mt=Nt;
nII=[-Mt/2+1:Mt/2]';
hnII=sin(2*pi*nII*Fc-(pi*Fc))./(nII*pi-(pi/2));
hnIIs=fft(hnII,FFT_size);
hrIIs=abs(hnIIs);

%Type 3 (hn odd):
Mt=Nt+1;
nIII=n;
hnIII=(cos(2*pi*nIII*Fc)-1)./(nIII*pi);
hnIII(M+1)=0;
hnIIIs=fft(hnIII,FFT_size);
hrIIIs=abs(hnIIIs);

%Type 4 (hn even);
Mt=Nt;
nIV=nII;
hnIV=2*(cos(2*pi*nIV*Fc-pi*Fc)-1) ./ (2*nIV*pi-pi);
hnIVs=fft(hnIV,FFT_size);
hrIVs=abs(hnIVs);
subplot(2,1,1); plot([hnI,[hnII;0], hnIII, [hnIV;0]]); grid;3
legend('1','2','3','4');
subplot(2,1,2); plot(frequency_vec,[hrIs,hrIIs,hrIIIs,hrIVs]);
set(gca,'xlim',[0,0.5]); grid;
legend('1','2','3','4');






