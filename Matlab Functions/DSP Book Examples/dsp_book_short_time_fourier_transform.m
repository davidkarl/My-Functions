function [spec,normtm]=tfct(xt,Lb,ovlp,Lfft,win)
%dsp book short term fourier transform
% xt=signal
% Lb=block size
% ovlp=overlap length
% Lfft=fft length
% win=window type
% spec=spectrogram
% normtm=time vector (normalized)


if nargin<4, win='rect';end
xt=xt(:);
x=xt;
Nx=length(xt);
if win=='hamm'
    wn=0.54-0.46*cos(2*pi*[0:Lb-1]'/Lb);
elseif win=='hann'
    wn=0.5-0.5*cos(2*pi*[0:Lb-1]'/Lb);
else
   wn=ones(Lb,1); 
end
blkS=(Lb-ovlp);
nbfen=floor(Nx/blkS);
Lxb=nbfen*blkS;

%calculating the index:
idxH=[1:Lxb];
idxtab=reshape(idxH,blkS,nbfen);
indx=idxtab(blkS,:)+1;
idxv=[1:ovlp-1]'*ones(1,nbfen);
indh=ones(ovlp-1,1)*indx;
idxtab2=[indx;idxv+idxh];
idxtab=[idxtab;idxtab2];
idxmax=idxtab(Lb,nbfen);
idlm=find(idxtab(Lb,:)>=Nx);
nbf=idlm(1);
xx=zeros(Lb,nbf);
x=[x;zeros(idxmax-Nx,1)];
xx(:)=x(idxtab(:,1:nbf));
Nc=size(xx,2);
xxp=xx.*(wn*ones(1,Nc));
spec=fft(xxp,Lfft);
normtm=[0:Nc-1]*blkS;
return




