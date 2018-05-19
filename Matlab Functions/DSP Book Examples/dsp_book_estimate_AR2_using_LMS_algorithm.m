%dsp book estimate AR2 process using LMS
a=[1,-1.5,0.8]';
P=length(a)-1;
N=4000;
w=randn(N,1); %sigma2w=1
x=filter(1,a,w); %signal
%estimate the eigenvalues:
r0=x'*x/N;
r1=x(2:N)'*x(1:N-1)/N;
r2=x(3:N)'*x(1:N-2)/N; %or D2=[x',0;0,x']
Rx=toeplitz([r0,r1]);
lambda=eig(Rx);
muMax=2/max(lambda); %estimated max
 
%LMS implementation:
mu=0.005;
gest=zeros(P,1);
err=[];
for n=P+1:N
   y=x(n-1:-1:n-P);
   en0=x(n)-gest'*y;
   err=[err;en0];
   gest=gest+mu*en0*y;
end
aest=[1;-gest];
%Yule-Walker equation:
aYule = [1;-inv(Rx)*[r1;r2]];
%display the results:
[a,aest,aYule]
Jlim=std(err)^2;
sigma2w_estime=r0+aYule(2)*r1+aYule(3)*r2;







