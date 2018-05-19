%dsp book test psola
[x,Fs]=wavread('rodena blabla.wav');
x=x(:,1);
gamma=0.5;
x_m=psola(x,Fs,gamma);
soundsc(x_m,Fs);


