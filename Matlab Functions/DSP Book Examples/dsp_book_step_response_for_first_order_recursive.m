%dsp book check step response for first order AR1
N=30;
mtime=(0:N-1);
a=[-2/3,1/2,3/4,7/8];
Na=length(a);
indic=ones(N,1);
y=zeros(N,Na);
for ii=1:Na
   y(:,ii)=filter(1-a(ii),[1,-a(ii)],indic); %i think they mixed up and it's filter(1,[1,-a(ii)],indic)
end
plot(mtime,y,'-',mtime,y,'o');
set(gca,'xlim',[0,N-1]);
set(gca,'ylim',[0,1.8]);
grid




