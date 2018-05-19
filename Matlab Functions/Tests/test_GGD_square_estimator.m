%test GGD square estimator:

apriori_SNR_dB = [-40:40];
aposteriori_SNR_dB = [-40:40];
gain = zeros(length(apriori_SNR_dB),length(aposteriori_SNR_dB));

for k=1:length(apriori_SNR_dB);
    for t=1:length(aposteriori_SNR_dB)

        apriori_SNR = 10^(apriori_SNR_dB(k)/10);
        aposteriori_SNR = 10^(aposteriori_SNR_dB(t)/10);
        GGD_nu = 10^(-2/10);
        GGD_rho = 2;
        GGD_tau = 10^(-3/10);
        
        mu_k = apriori_SNR.*aposteriori_SNR./(GGD_nu+apriori_SNR);
        
        apriori_SNR_numinator = ConflHyperGeomFun((GGD_nu+GGD_rho-0.5)/2,1/2,mu_k.^2/2)/gamma((GGD_nu+GGD_rho+0.5)/2) ...
            + sqrt(2)*mu_k.*ConflHyperGeomFun((GGD_nu+GGD_rho+0.5)/2,3/2,mu_k.^2/2)/gamma((GGD_nu+GGD_rho-0.5)/2);
        
        apriori_SNR_denominator = ConflHyperGeomFun((GGD_nu-0.5)/2,1/2,mu_k.^2/2)/gamma((GGD_nu+0.5)/2) ...
            + sqrt(2)*mu_k.*ConflHyperGeomFun((GGD_nu+0.5)/2,3/2,mu_k.^2/2)/gamma((GGD_nu-0.5)/2);
        
        apriori_SNR = 1/4./aposteriori_SNR * gamma(GGD_nu+GGD_rho-0.5)/gamma(GGD_nu-0.5) .* (apriori_SNR_numinator./apriori_SNR_denominator);

        gain(k,t) = apriori_SNR;
    end
end

imagesc(flip(apriori_SNR_dB),flip(aposteriori_SNR_dB),gain);
title(strcat('number of nans = ',num2str(sum(isnan(gain(:))))));
