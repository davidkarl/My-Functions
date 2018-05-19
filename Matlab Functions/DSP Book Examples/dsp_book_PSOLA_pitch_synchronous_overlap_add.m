function s_synt = psola(s_orig,Fs,gamma)
%dsp book PSOLA (pitch synchoronous overlap add
%s_orig=signal
%Fs=sampling frequency(Hz)
%gamma=modification rate
%s_Synt=modified signal

seuil_pitch=0.7;
Rsurech=1; %to improve the pitch's evaluation
L10ms=fix(Fs/100); %constant size window (10ms)
fp_min=70;
fp_max=400;
Lfen=2*fix(Fs/fp_min);

Ns=length(s_orig);
Namax=fix(Ns*fp_max/Fs);
ta=zeros(Namax,1);
ta(1)=1;
Pa=L10ms;
inda=1;

%analysis
while ta(inda)<Ns-Lfen
   indsdeb=ta(inda);
   %the length Lfen must be large enough to allow the estimation of the
   %lowest frequency
   indsfin=indsdeb+Lfen;
   sextraint=s_orig(indsdeb:indsfin);
   [Fpitch,corr]=f0cor(sextrait,Fs,Rsurech,seuil_pitch,fp_min,fp_max);
   if isnan(Fpitch)
      Pa=L10ms; 
   else
      Pa=fix(Fs/Fpitch);
   end
   inda=inda+1;
   ta(inda)=ta(inda-1)+Pa;
end
ta=ta(1:inda);
Na=length(ta);

%time scale modification and synthesis:
s_synt=zeros(fix(Ns/gamma),1);
ii=1;
ts=1;
ie=1;
te=1;
while ie<Na-2
   ii=ii+1;
   te=te+gamma;
   ie=ceil(te);
   Pa=ta(ie+1)-ta(ie);
   ts=ts+Pa;
   winHann=sin(pi*(0:2*Pa)'/(2*Pa)).^2;
   sola=s_orig(ta(ie):ta(ie)+2*Pa).*winHann;
   s_synt(ts-Pa:ts+Pa)=s_synt(ts-Pa:ts+Pa)+sola;
end

return







