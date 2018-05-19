tic
load tab_cod %tab_cod(nblocs,XX);
excgl=eye(1,40); %glottal signal
lbloc=240; %block length
recouv=50; %overalp
ltr=lbloc-recouv;
OvlRec=lbloc/3; %overlap reconstruction 1/3
LBrec=lbloc+2*(OvlRec-recouv); %reconstructed block length
nblocs=size(tab_cod,1);
NbParam=size(tab_cod,2);
outsig=[];
finalsig=zeros(1,nblocs*ltr+OvlRec);

%reconstruction window:
fen_Rec=[(1:OvlRec)/OvlRec, ones(1,lbloc-2*recouv),(OvlRec:-1:1)/OvlRec];
ImpGlPtr=0;
LgExcGl=length(excgl);
NbSmpTot=LBrec+LgExcGl; %because of filtering
drap_vnv=0;

for k=2:nblocs-1
   if tab_cod(k,2)~=0 %voiced block
       if drap_vnv==1 %the previous one is voiced
            %continuity of the input signal
            trame=[TmpSig(ltr+1:NbSmpTot),zeros(1,ltr)];
            NbSmp=NbSmpTot-ltr+ImpGlPtr;
       else %the previous one is not voiced
           trame=zeros(1,NbSmpTot);
           NbSmp=0;
       end
       PitchPeriod=tab_cod(k,2); %block pitch
       
       while NbSmp<LBrec
          trame((NbSmp+1):(NbSmp+LgExcGl))=excgl;
          NbSmp=NbSmp+PitchPeriod;
       end
       drap_vnv=1;
       ImpGlPtr=NbSmp-NbSmpTot;
       TmpSig=trame;
       trame=trame(1:LBrec);
       trame=trame/std(trame); %normalization
       
   else %not voiced
       ImpGlPtr=0; 
       drap_vnv=0; 
       trame=randn(1,LBrec); %gaussian white noise
   end
   
   trame=sqrt(tab_cod(k,1))*trame; %power
   %den=ki2ai(tab_cod(k,3:NbParam));
   den=[1,tab_cod(k,3:NbParam)];
   outsig=filter(1,den,trame);
   outsig=fen_rec.*outsig;
   st=(k-1)*ltr;
   
   %construction with an overlap:
   finalsig((st+1):(st+LBrec))=finalsig((st+1):(st+LBrec))+outsig;
   
end

finalsig=filter(1,[1,-0.9375],finalsig); %de emphasis
toc
soundsc(finalsig,8000);



