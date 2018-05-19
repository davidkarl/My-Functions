function [beta] = audio_berouti_beta2(SNR)

if SNR>=-5.0 && SNR<=20
   beta=4-SNR*3/20; 
else
   
  if SNR<-5.0
   beta=5;
  end

  if SNR>20
    beta=1;
  end
  
end