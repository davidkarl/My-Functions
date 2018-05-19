% Example 7.6
% Computational Statistics Handbook with MATLAB, 2nd Edition
% Wendy L. and Angel R. Martinez

%Load up the data:
load mcdata
number_of_samples = length(mcdata);  								

%Null and Alternative hypotheses:
%H0: mu=454
%H1: mu<454

%Population sigma is known.
sigma = 7.8;					
xbar_sigma = sigma/sqrt(number_of_samples);

%Get the observed value of the test statistic:
Tobs = (mean(mcdata)-454)/xbar_sigma; %here the test statistics is (xbar-454)/xbar_sigma
Tobs1 = mean(mcdata); %here the test statistics is the mean

%This command generates the normal probability plot.
%It is a function in the MATLAB Statistics Toolbox.
normplot(mcdata)
%we can see from this normality plot that the population can be considered
%as drawn from a normal distribution. so we can use this model when drawing
%samples in the monte carlo trials.

number_of_monte_carlo_trials = 1000;				% Number of Monte Carlo trials
%Storage for test statistics from the MC trials.
Tm = zeros(1,number_of_monte_carlo_trials);
%Start the simulation.
for i = 1:number_of_monte_carlo_trials
    %Generate a random sample under H_0
    xs = sigma*randn(1,number_of_samples) + 454;
    
    %Get current test statistic
    Tm(i) = (mean(xs) - 454)/xbar_sigma;
    Tm1(i) = mean(xs);
end

% Get the critical value for alpha.
% This is a lower-tail test, so it is the
% alpha quantile.
alpha = 0.05;
cv = csquantiles(Tm,alpha);


%Get the p value. this is a lower tail test. find all the values from the
%simulation that are below the observed value of the test statistics.
ind = find(Tm1<=Tobs1);
p_value_estimate = length(ind)/number_of_monte_carlo_trials;










