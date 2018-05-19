%dsp book analyze and synthesize signal with several harmonics
figure(1);
load piano;
Fc=24000;
N=length(piano);

%splitting in blocks:
lbloc=350;
nbblocs=fix(N/lbloc);
pianoF=piano(1:nbblocs*lbloc);
xsyn=zeros(nbblocs*lbloc,1);
tpsbloc=(0:lbloc-1)/Fe;

%windows:
fenH=0.54-0.46*cos(2*pi*(0:lbloc-1)'/(lbloc-1));
fenH=fenH*lbloc/sum(fenH); %normalization
fenT=2*[(0:lbloc/2-1)';(lbloc/2-1:-1:0)']/lbloc;

%parameters of the spectral analysis:
P=12;
Lfft=4096;
deltaf=2*round(Lfft/lbloc);
fq=Fe*(0:Lfft/2-1)/Lfft;

%processing:
for jj=0:2*nbblocs-2
   jj1=(lbloc/2)*jj+1;
   jj2=jj1+lbloc-1;
   x=pianoF(jj1:jj2).*fenH;
   x=x-mean(x);
   fs=zeros(1,P);
   mm=zeros(P,1);
   
   %spectrum:
   xf=fft(x,Lfft);
   xf=xf(1:Lfft/2)/lbloc;
   xfvar=f;
   xfvar(1:deltaf)=zeros(1,deltaf);
   
   %analysis:
   for ii=1:P
      [bid,im]=max(abs(xfvar));
      fs(ii)=(im-1)/Lfft;
      mm(ii,1)=xfvar(im);
      u1=max(1,im-deltaf);
      u2=min(Lfft/2,im+deltaf);
      nb=u2-u1+1;
      xfvar(u1:u2)=zeros(1,nb);
   end
   
   %synthesis:
   xsyn_f=2*real(exp(2*pi*1i*(0:lbloc-1)'*fs)*mm);
   %overlap-add:
   xsyn(jj1:jj2)=xsyn(jj1:jj2)+xsyn_f.*fenT;
   subplot(2,1,1);
   plot(tpsbloc,pianoF(jj1:jj2),':',tpsbloc,xsyn_f); grid
   
   %drawing the spectra
   subplot(2,1,2);
   plot(fq,20*log10(abs(xf)));
   set(gca,'ylim',[-70,0]);
   hold on;
   plot(fs*Fe,20*log10(abs(mm)),'or'); hold off;
   grid;
   pause;
end 
    
ti=(0:nbblocs*lblock-1);
%displaying reconstructed signal:
figure(2);
plot(ti,pianoF,'b',ti,xsyn,'r');
grid


    
end






