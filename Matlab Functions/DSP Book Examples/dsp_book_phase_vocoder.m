
[x,Fe]=wavread('rodena blabla.wav');
gamma=0.5;
Lfft=256;
x_m=phasevoc(x,gamma,Lfft);
soundsc(x_m,Fe);





 