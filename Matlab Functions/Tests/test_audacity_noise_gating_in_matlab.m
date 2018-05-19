%test audacity noise gating in matlab

clc;
clear;
Fs = 8000;
starting = 1;
ending = 8000;
load('y.mat');%contains the noisy signal 

windowSize = 2048;
noiseGate = zeros(windowSize, 1);
sum = zeros(windowSize, 1);
sumsq = zeros(windowSize, 1);
profileCount = zeros(windowSize, 1);
smoothing = zeros(windowSize, 1);
level = 8;

portion = y(starting:ending);%gets the noise part to build a noise profile

%Noise Profile:
for i=1:floor(length(portion)/windowSize)
    ind = ((i-1)*windowSize)+1;
    in = portion(ind:ind+windowSize-1);

    out = PowerSpectrum(windowSize, in);

    for j=1:windowSize/2
        value = log(out(j));
        if(value ~= inf)
            sum(j) = sum(j)+value;
            sumsq(j) = sumsq(j)+value^2;
            profileCount(j) = profileCount(j)+1;
        end
    end
end
for i=1:windowSize/2 + 1
    noiseGate(i) = sum(i)/profileCount(i);
end


%Noise Removal:
for i=1:(windowSize/2):length(y)-windowSize+1
    %ind = ((i-1)*windowSize)+1;
    inr = y(i:i+windowSize-1);

    infft = conj(fft(inr));

    outr = real(infft);
    outi = imag(infft);

    inr = WindowFunc(3, windowSize, inr);
    power = PowerSpectrum(windowSize, inr);

    pLog = zeros((windowSize/2)+1, 1);
    for j=1:(windowSize/2)+1
        pLog(j) = log(power(j));
    end

    half = windowSize/2;
    for j=1:half+1;
        if(pLog(j) < noiseGate(j) + (level/2))
            smooth = 0.0;
        else
            smooth = 1.0;
        end
        smoothing(j) = smooth*0.5 + smoothing(j)*0.5;
    end

   for j=3:half-1
      if (smoothing(j)>=0.5 && smoothing(j)<=0.55 && smoothing(j-1)<0.1 && smoothing(j-2)<0.1 && smoothing(j+1)<0.1 && smoothing(j+2)<0.1)
          smoothing(j) = 0.0;
      end
   end

   outr(1) = outr(1)*smoothing(1);
   outi(1) = outi(1)*smoothing(1);
   outr(half+1) = outr(half+1)*smoothing(half+1);
   outi(half+1) = outi(half+1)*smoothing(half+1);

   for j=2:half
       k = windowSize - (j-2);
       smooth = smoothing(j);
       outr(j) = outr(j)*smooth;
       outi(j) = outi(j)*smooth;
       outr(k) = outr(k)*smooth;
       outi(k) = outi(k)*smooth;     
   end

   outTmp = outr - 1j.*outi;
   inr = real(ifft(conj(outTmp)));
   inr = WindowFunc(3, windowSize, inr);

   yFF(i:i+(windowSize/2)-1) = inr(1:windowSize/2);

end

 sound(yFF, 8000);
 
 
 
 
 