function [beta] = audio_berouti_beta1(SNR)


if SNR>=-5.0 && SNR<=20
    beta=3-SNR*2/20;
else
    
    if SNR<-5.0
        beta=4;
    end
    
    if SNR>20
        beta=1;
    end
end
    
