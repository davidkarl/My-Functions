%dsp echo cancelation
N=4000;
alpha=0.2;
yn=randn(N,1); %reference
hh=[1,0.3,-0.1,0.2];
xn=filter(hh,1,yn); %echo

%LMS implementation:
mu=0.05;
P=20;
gn=zeros(P,1);
en=zeros(N,1);

for n=P:N
   en0=xn(n)-gn'*yn(n:-1:n-P+1);
   gn=gn+mu*en0*yn(n:-1:n-P+1);
   en(n)=(1-alpha)*en(n-1)+alpha*abs(en0)^2;
end
%displaying the results
plot(10*log10(en(P+1:N)));
grid;
set(gca,'xlim',[0,3000]);
%plot(20*log10(abs(fft(hest,1024))))



%ECHO cancellation 2:
load phrase
sn=sn(:);
N=length(sn);
mm=max(abs(sn));
sn=sn/mm;
yn=randn(N,1); %reference
hh=[1,0.3,-0.1,0.2];
echo=filter(hh,1,yn);
xn=sn+echo;
subplot(3,1,1); plot(sn); grid; set(gca,'xlim',[3500,6800]);
subplot(3,1,2); plot(xn); grid; set(gca,'xlim',[3500,6800]);

%implementing the LMS:
mu=0.01;
P=20;
gn=zeros(P,1);
en=zeros(N,1); %denoised signal
for n=P:N
   en0=xn(n)-gn'*yn(n:-1:n-P+1);
   gn=gn+mu*en0*yn(n:-1:n-P+1);
   en(n)=en0;
end


%displaying results:
subplot(3,1,3); plot(en); grid;
set(gca,'xlim',[3500,6800]);

% %Audio tests:
% soundsc(sn,Fs);
% soundsc(xn,Fs);
% sound(en,Fs);



