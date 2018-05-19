% Example 7.3
% Computational Statistics Handbook with MATLAB, 2nd Edition
% Wendy L. and Angel R. Martinez

% Get several values for the mean under the alternative
% hypothesis. Note that we are getting some values
% below the null hypothesis.
mu_vec = 40:60;
true_mu = 45;

%Note the critical value:
cv_critical_value = 1.645;
%Note the standard deviation for x-bar:
sigma_standard_deviation_for_xbar = 1.5;
%It's easier to use the unstandardized version, so convert:
xbar_95_percent_probability_to_be_below_of_value = true_mu + cv_critical_value*sigma_standard_deviation_for_xbar;

%Get a vector of ct values that is the same size as mualt:
xbar_95_percent_probability_to_be_below_of_value_vec = xbar_95_percent_probability_to_be_below_of_value*ones(size(mu_vec));
%Now get the probabilities to the left of this value. These are the probabilities of the Type II error:

%OUR TEST IS WHETHER T IS SMALLER THEN SOME VALUE, IF IT'S LARGER WE REJECT(!!) THE NULL HYPOTHESIS.
%THUS OUR PROBABILITY OF TYPE II ERROR FOR EVERY mu IS THE PROBABILITY, UNDER THAT HYPOTHESIS, 
%TO BE BELOW A CERTAIN VALUE!!!

%beta = probability of type II error:
beta = normcdf(xbar_95_percent_probability_to_be_below_of_value_vec,mu_vec,sigma_standard_deviation_for_xbar);
%To get the power 1-beta
pow = 1 - beta;
%Plot beta
plot(mu_vec,pow);
xlabel('True Mean \mu')
% ylabel('Probability of Type II Error - \beta')
ylabel('test power - (1-\beta)');
axis([40 60 0 1.1])


%To get the p-value (observed significance test). the p value is defined as
%the probability of observing a value of the test statistics as extreme as
%or more extreme than the one that is observed, when the null hypothesis H0
%is true. the word extreme refers to the direction of the alternative
%hypothesis. for example, if a small value of the test statistics (a lower
%tail test) indicates evidence for the alternative hypothesis, then the p
%value is calculated as P_H0(T<=t0), where t0 is the observed value of the
%test statistics T, and P_H0 denotes the probability under the null
%hypothesis. 
z_obs = (xbar_95_percent_probability_to_be_below_of_value-true_mu)/sigma_standard_deviation_for_xbar;
p_value = 1-normcdf(z_obs,0,1); %lower tail test


%Confidence intervals:
mu = 45;
sigma_standard_deviation_per_sample = 15;
number_of_samples = 100;
alpha = 0.05;
xbar = 47.2;

%Get the 95% confidence interval.
%Get the value for z_alpha/2
zlo = norminv(1-alpha/2,0,1);
zhi = norminv(alpha/2,0,1);
theta_lower_limit = xbar - zlo*sigma_standard_deviation_per_sample/sqrt(number_of_samples);
theta_upper_limit = xbar - zhi*sigma_standard_deviation_per_sample/sqrt(number_of_samples);


1;




