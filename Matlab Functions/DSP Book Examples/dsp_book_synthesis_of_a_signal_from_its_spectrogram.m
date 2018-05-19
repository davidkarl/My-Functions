function sig=spec2si(spec,n0,win)
%dsp book synthesis of a signal from its spectrogram
%spec=spectrogram
%n0=shift value
%sig=signal
[Lfft,nbcol]=size(spec);
ispec=real(ifft(spec));
sig=zeros(Lfft+(nbcol-1)*n0,1);
%resynthesis using a window:
for icol=1:nbcol
   sigfen=ispec(:,icol).*win;
   %overlap add:
   ixi=(icol-1)*n0+1;
   sig(ixi:ixi+Lfft-1)=sig(ixi:ixi+Lfft-1)+sigfen;
end
return



