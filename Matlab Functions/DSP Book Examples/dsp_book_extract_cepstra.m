function cepstre=extractCEPSTRE(xt,Fe)
%dsp book exctract cepstra
%xt =audio signal
%Fe=sampling frequency
%cepstre = cepstral coefficients
xt=xt(:);

%parameters
pp=10; %cepstrum order
duree=15; %window duration in ms
Lfen=fix(Fe*duree/1000); %window size
deal=fix(Lfen/2); %shift for overlapping
Lfft=2^nextpow2(Lfen); %FFT size
hamm=0.5-0.5*cos(2*pi*(0:Lfen-1)'/Lfen);
Lx=length(xt);
nbfen=fix(Lx/decal);
cepstre=zeros(pp,nbfen);

for ii=1:nbfen-1
   inddeb=(ii-1)*deal+1;
   indfin=inddeb+Lfen-1;
   xaux=xt(inddeb:indfin).*hamm;
   
   %compute standard cepstral coefficients:
   Sx=log(abs(fft(xaux,Lfft)));
   
   %power cepstrum:
   Cx=real(ifft(Sx));
   cepstre(:,ii)=Cx(2:pp+1); %without energy
end









