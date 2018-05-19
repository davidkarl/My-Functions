%dsp book polyphase FIR filtering
x0=[1:103];
lx0=length(x0); M=4;
b=0.3; 
N=25; 
h=rif(N,b);

%M-polyphase filters (with insertion of zeros for the processing):
z1=filter(h,1,x0);

%polyphase processing:
z2=zeros(M,lx0);
for k=1:M
   xx = [zeros(1,k-1), x0(1:lx0-k+1)];
   z2(k,:)=filter(hp(k,:),1,xx);
end

xx=sum(z2);
[xx(1:lx0)', z1(1:lx0)']



