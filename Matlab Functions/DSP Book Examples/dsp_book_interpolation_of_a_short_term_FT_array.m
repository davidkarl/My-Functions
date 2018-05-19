function spec_s=specinterp(spec_a,gamma)
%dsp book interpolation of a short term FT array
%spec_a = original spectrogram (short term FT)
% gamma = temporal modification rate
%spec_s = modified spectrogram

[Lfft,nbcol]=size(spec_a);
ts=1:gamma:nbcol-1;
spec_s=zeros(Lfft,length(ts));

%phase and phase increase
phase_a=angle(spec_a);
modula_a=abs(spec_a);
diffp=zeros(Lfft,1);
phase_s=phase_a(:,1);
indcol=1;

for tt=ts
   %two adjacent columns
   ta_min=floor(tt);
   ta_max=floor(tt)+1;
   %weighted mean:
   pond=tt-floor(tt);
   modul=(1-pond)*module_a(:,ta_min)+pond*module_a(:,ta_max);
   spec_s(:,indcol)=modul.*exp(1i*phase_s);
   %phase diff and accumulation:
   diffp=phase_a(:,ta_max)-phase_a(:,ta_min);
   phase_s=phase_s+diffp;
   indcol=indcol+1; 
end
return










