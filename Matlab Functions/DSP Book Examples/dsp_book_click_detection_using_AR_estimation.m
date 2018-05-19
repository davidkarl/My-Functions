%dsp book click detection using AR estimation
%original signal (order 10-AR)
a=[1,-1.6507,0.6711,-0.1807,0.6130,-0.6085,0.3977,-0.611,0.5412,0.1321,-0.2393];
K=length(a);
N=500;
w=randn(1,N);
s=filter(1,a,w);
srms=sqrt(s*s'/N);

%NBCRAC clicks with an amplitude +-1.5 srms
nbcrac=5;
poscrac=[73,193,249,293,422];
ampcrac=1.5*srms*(2*round(rand(1,nbcrac))-1);
sig=s;
sig(poscrac)=s(poscrac)+ampcrac;
subplot(3,1,1); plot(s); grid
subplot(3,1,2); plot(sig); grid;

%detection of the clicks:
[aest,sw2est]=xtoa(sig,K); %estimation of the AR
y=filter(aest,1,sig); %whitening: estimation of the residual
z=filter(aest(K:-1:1),1,y); %matched filtering
subplot(3,1,3); plot(z); grid
V0eff=sqrt(sw2est*aest'*aest);
lambda=3;
threshold=lambda*V0eff;
izthreshold=find(abs(z)>threshold); %threshold
izthreshold=izthreshold-K; %filter delay
lzs=length(izthreshold);

%extraction of the maxima (3 samples from each other)
dis=izthreshold-[0,izthreshold(1:lzs-1)];
mpl3=find(dist>3);
lm3=length(mpl3);
mpl3=[mpl3,lzs+1];

for ii=1:lm3
   t1=izthreshold(mpl3(ii));
   t2=izthreshold(mpl3(ii+1)-1);
   [zmax(ii),im]=max(z(t1:t2));
   posEstim(ii)=im+t1;
end
izthreshold,poscract,posEstim










