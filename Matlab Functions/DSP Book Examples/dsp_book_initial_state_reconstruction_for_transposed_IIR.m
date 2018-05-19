function zi=filtricII(num,den,xinp,xout)
%dsp book reconstruction of initial state for transposed IIR structure
%num = [b0,...,bp]
%den = [1,a1,...ap]
% xinp=input sequence
% xout=output sequence
% zi=reconstructed initial state

lden=length(den);
lnum=length(num);
if lden<lnum, den(lnum)=0; lden=lnum; end
if lnum<lden, num(lden)=0; end

ld=lden-1;
numv=zeros(lden,1);
denv=numv;
numv(:)=num;
denv(:)=den;

lx=length(xinp);
ly=length(xout);
if lx<ld, xinp(ld)=0; end
if ly<ld, xout(ld)=0; end
ysv=zeros(1,ld);
xev=ysv;
ysv(:)=xout(1:ld);
xev(:)=xinp(1:ld);
zi=filtrerII(denv,1,ysv)+filtrerII(-numv,1,xev);
return






