%test MA process modeling using AR
N=3000;
b1=[1;-0.7];
sigmaw2=1;
w=sigmaw2*randn(N,1);
x=filter(b1,1,w);
Lfft=1024;
b1s=fft(b1,Lfft);
fq=(0:Lfft-1)/Lfft;
Sth=sigmaw2*abs(b1s).^2;

%welch method:
Swelch=welch(x,16,'h',Lfft,0.95);
%K=2 covariances
Scov2f=covtodsp(x,2,'b',Lfft).';
%K=4 covariances:
Scov4f=covtodsp(x,4,'b',Lfft).';
%Durbin method:
[b1ch,sigma2ch]=durbin(x,1,15);
Sdurb=sigma2ch*abs(fft(b1ch,Lfft)).^2;
plot(fq,[Sth,Swelch,Scov2f,Scov4f,Sdurb]);
legend('theoretical','welch','psd from covariance 2 terms','psd from covariance 4 terms','durbin');
set(gca,'xlim',[0,0.5]); grid


