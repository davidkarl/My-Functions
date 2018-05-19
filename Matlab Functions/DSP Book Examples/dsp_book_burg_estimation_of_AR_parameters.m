N=20000;
w=randn(N,1)+3*1i*randn(N,1);
avrai=[1;0.48-0.45*1i;0.89-0.22*1i;0.48-0.4*1i;-0.01-0.22*1i];
P=length(avrai)-1;
x=filter(1,avrai,w);
K=7; %over estimated order if K>P

%estimation using burg:
[aestB,sest2B]=burg(x,K);
%estimation using levinson:
[aestL,sest2L]=levinson(x,K);
%direct estimation:
[aestD,sest2D]=xtoa(x,K);

[[avrai;zeros(K-P,1)], aestD, aestL(:,K+1), aestB]













