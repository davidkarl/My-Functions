function zi=filteric(num,den,xi,y0)
%dsp book initial state reconstruction for a direct canonical structre

%num = [b0,...bp]
%den = [1,a1,...ap]
%xi = input sequence
%y0 = output sequence
%zi = reconstructed initial state

lden=length(den);
lnum=length(num);
if (lden<lnum), den(lnum)=0; lden=lnum; end
if (lnum<lden), num(lden)=0; end

ld=lden-1;
numv=zeros(lden,1);
denv=numv;
numv(:)=num;
denv(:)=den;
lx=length(xi); ly=length(y0);

if lx<ld, xi(ld)=0; end
if ly<ld, yo(ld)=0; end

ysv=zeros(1,ld); 
xev=zeros(1,ld);
ysv(:)=yo(ld:-1:1); 
xev(:) = xi(ld:-1:1);
x=[ysv;xev];
vec=zeros(2*ld,1);
vec(:)=x;
vo=[numv;zeros(ld-1,1); denv; zeros(ld,1)];
A=[];
for ii=1:ld, A=[A,v0]; end
A=A(1:4*ld*ld);
Ax = zeros(2*ld,2*ld); 
Ax(:)=A;
Ax=Ax';
zzi=inv(Ax)*vec;
zi=zzi(ld+1:2*ld);
return



