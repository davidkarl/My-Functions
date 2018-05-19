%dsp book coding speech based on an AR model:
% OUTPUT: array tab_cod(N,xx):
% tab_cod(N,1): energy in the block of signal
% tab_cod(N,2): pitch period
% tab_cod(N,3:12) AR coefficients (AR-ordv if voiced sound, AR-ordnv otherwise) or reflection coefficients
% each bock has 240 samples length(30ms) with 60 samples overlap
load phrase;
enerm=std(y)^2*0.1;
%AR model orders for voiced and non voiced sounds
ordv=20;
ordnv=10;
NbParam=ordv+2;
phrase=y-mean(y);

%parameters:
lbloc=240; %block length
recouv=60; %overlap
ltr=lbloc-recouv; 
nblocs=floor((length(phrase)-recouv)/ltr); %Nb of blocks
phrase=phrase(1:length(phrase)-reste);
tmin=40;
tmax=150;
seuil=0.7; %for pitch detection

vnv=zeros(1,nblocs); %boolean voiced/non voiced
pitch=zeros(1,nblocs); %pitch period
tab_cod=zeros(nblocs,NbParam); %coeffts of the model

%detection voiced/non voiced:
sprintf('"voiced/non voiced" on %5.0f blocks',nblocs)

tic
for k=1:nblocs
   ind=(k-1)*ltr;
   blocan=phrase(ind+1:ind+lbloc); %analysis block
   [vnv(k),pitch(k)]=detectpitch(blocan,seuil,tmin,tmax,enerm);
end
toc

%AR model:
sprintf('AR-model');
tic
preacpar=filter([1,-0.9375],1,phrase); %pre-emphasis
for k=2:(nblocs-1)
   if (vnv(k-1)==vnv(k+1)) %correction of errors of detection
       vnv(k)=vnv(k-1);
       if vnv(k)==1
          %voiced with pitch=mean
          pitch(k)=floor((pitch(k-1)+pitch(k+1))/2);
       else
          %non voiced with pitch 0
          pitch(k)=0;
       end
   end
   
   %analysis block:
   sigblock=preapar((k-1)*ltr+1:(k-1)*ltr+lbloc);
   if vnv(k)==1
      [pcoeff,enrg]=xtoa(sigbloc,ordv);
      %coeff_refl=ai2ki(pcoeff); %reflection coeffts
      %tab_cod(k,3:NbParam)=coeff_refl; %coeffts
      tab_cod(k,3:NbParam)=pcoeff(2:ordv+1)';
      tab_cd(k,1)=enrg;
      tab_cod(k,2)=pitch(k);
   else
       [pcoeff,enrg]=xtoa(sigbloc,ordnv);
       %coeff_refl=ai2ki(pcoeff); %reflection coeffts
       %tab_cod(k,3:NbParam)=coeff_refl;
       tab_cod(k,1)=enrg;
       tab_cod(k,2)=0;
       tab_cod(k,3:NbParam)=[pcoeff(2:ordnv+1)',zeros(1,ordv-ordvn)];
   end
end
toc
sprintf('writing array in tab_cod.mat');
save tab_cod tab_cod







