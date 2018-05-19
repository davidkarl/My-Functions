%dsp book polyphase undersampling and oversampling
M=4;
N=1500;
L=16; %N and M must be multiple of M
x=randn(N,1);
h=(1:L);
%direct undersampling -> yu:
y=filter(h,1,x);
yu=y(1:M:N);
%parallelized undersampling -> yp:
yp=zeros(N/M,1);
for k=1:M
   auxx=x(k+1:M:N);
   lx=length(auxx);
   auxh=h(M-k+1:M:end);
   yp(1:lx)=yp(1:lx)+filter(auxh,1,auxx);
end
max(abs(yu(M+1:lx)-yp(M:lx-1)))


M=4;
N=150;
L=16;
x=randn(N,1);
h=(1:L);
xo=zeros(N*M,1);
xo(1:M:end)=x;
%direct oversampling -> yo:
yo=filter(h,1,xo);
yp=zeros(N*M,1);
%parallelized oversampling -> yp:
for k=1:M
   auxh=h(k:M:end);
   yp(k:M:N*M)=yp(k:M:N*M)+filter(auxh,1,x);
end
max(abs(yo(M:end)-yp(M:end)))


