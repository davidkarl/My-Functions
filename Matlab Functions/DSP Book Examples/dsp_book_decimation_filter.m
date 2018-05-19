function y=decM(x,M,Nf)
%dsp book decimation filter
%x=input sequence
%M=decimation ratio
%Nf=2*Nf+1 coeffts filter
%y=output sequence
if nargin<3, Nf=20; end
theta=(1:Nf)*pi;
h=sin(theta/M).(theta/M);
h=h.*(0.54+0.46*cos(theta/Nf));
h=[fliplr(h),1,h]/M;
x0=zeros(length(x)+Nf,1);
x0(1:length(x))=x;
y=filter(h,1,x0);
y=y(Nf+1:M:length(y)); %decimation
return





