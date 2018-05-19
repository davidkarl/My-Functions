h=[1;0.3+1i;-0.2-6*1i;0.1;0.07;0.9];
lh=length(h);
%positive toeplitz matrix (lhXlh):
N=100; 
w=randn(N,1);
x=filter(0.5*ones(2,1),1,w);
for ii=1:lh
   cxx(ii)=x(1:N-ii+1)'*x(ii:N); 
end
RXX=toeplitz(cxx);
rxy=Rxx*h;
ha=invToeplitz(cxx,rxy); 
[h,ha]











