function [F0,corr]=f0cor(sn,Fe,R,thr_corr,Fmin,Fmax)
%dsp book fundamental frequency estimation:
% sn=signal from which the frequency is extracted
%Fe=sampling frequency
%R=oversampling factor
%thr_corr=threshold
%Fmin=min frequency (Hz) (otherwise Fmin=2*Fe/longueur(sn);
%Fmax=max frequency (Hz) (otherwise Fmax=Fe/2-Fmin
%corr=correlation sequence
sn=interp(sn,R);
Fe=R*Fe;
N=length(sn);
sn=sn(:);
sn=sn-mean(sn);
lagmin=fix(Fe/Fmax);
lagmax=fix(Fe/Fmin);
corr=zeros(1,lagmax-lagmin+1);
%the effects of the window size can be tested by taking wlg<wlgmax=N-lagmax
wlg=N-lagmax;
v0=sn(1:wlg);
for ii=lagmin:lagmax
   vP=sn(ii:ii+wlg-1);
   corr(ii-lagmin+1)=(v0'*vP)/sqrt((v0'*v0)*(vP'*vP));
end
[niv1,indmax]=max(corr);
if niv1<thr_corr
   pf0=0;
   F0=nan;
   return
else
   for ii=lagmin+1:lagmax
      if corr(ii-lagmin+1)>niv1*0.9
          while corr(ii-lagmin+1)>corr(ii-lagmin)
              ii=ii+1;
          end
          pf0=ii-2;
          F0=Fe/pf0;
          return;
      else
         F0=Fe/(indmax+lagmin-1); 
      end
   end
end
return




