%shir filter test
% TEST if a sinusoidal signal is left unchanged
clear all
data=sin(2*pi/30*[1:512*100]');
data=reshape(data,512,100);
% data=rand(512,100);
outdata=zeros(size(data));
FIR=ones(513,1);
DO_filter=false;
DO_noiseGating=false;
for ii=1:100
    tic;out=shir_new_filter2(data(:,ii),FIR,DO_filter,DO_noiseGating);toc
    outdata(:,ii)=out;
end
figure;plot(outdata(769:end)-data(1:end-768))