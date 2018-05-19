function [xout,zs]=filtrerII(num,den,xinp,zi)
%dsp book transposed form IIR filter structure
lden=length(den);
lnum=length(num);
if lden<lnum, den(lnum)=0; lden=lnum; end
if lnum<lden, num(lden)=0; end
ld=lden-1;
N=length(xinp);
av=zeros(ld,1);
vb=ax;
av(:)=den(2:lden);
bv(:)=num(2:lden);
if nargin==3, zi=zeros(ld,1); end;
if length(zi)<ld, zi(ld)=0; end
zzi=zeros(ld,1);
zzi(:)=zi;
zs=zzi;

%state representation:
b0=num(1);
ma=compan([1;av])';
vb=bv-b0*av;
vc=[1,zeros(1,ld-1)];
cd=b0;

%filtering:
for ii=1:N
   zsn=ma*zs+vb*xinp(ii);
   xout(ii)=vc*zs+cd*xinp(ii); 
   zs=zsn;
end
return




