function [ys,xs] = filterer(num,den,xe,xi)
%dsp book direct canonical structre filter

%num = [b0,b1,...bp]
%den = [1,a1,a2,...ap]
%xe = input sequence
%xi = initial state
%ys = output sequence
%xs = final state

lden=length(den);
lnum=length(num);
if (lden<lnum), den(lnum)=0; lden=lnum; end
if (lnum<lden), num(lden)=0; end

ld=lden-1;
N=length(xe);
av=zeros(1,ld);
bv=av;
av(:)=den(2:lden);
bv(:)=num(2:lden);

if (nargin==3), zzi=zeros(ld,1); end
if (nargin==4),
    if length(xi)<ld, xi(ld)=0; end
    zzi=zeros(ld,1);
    zzi(:)=xi;
end

for ii=1:N,
    x0n=xe(ii)-av*xs;
    ys(ii)=b0*x0n+vb*xs;
    xs=[x0n;xs(1:ld-1)]; %new state
end

return







