function [correct_classifications_vec,thresh] = statistics_get_ROC_curve_for_multivariate_normal_conditionals(...
                                                                        Z_input_target_class,...
                                                                        Z_input_non_target_class,...
                                                                        number_of_samples_for_cross_validation,...
                                                                        false_alarm_probabilities_vec)
% CSROCGEN	Generate a Receiver Operating Characteristic (ROC) curve.
%
%	[PCC, THRESH] = CSROCGEN(X2,X2,K,PFA)
%
%	This function generates a ROC curve, where the multivariate normal
%	is used to model the class-conditional probabilities. This will
%	generate the probability of correctly classifying the target
%	class (X1), given the desired false alarm rates (PFA). It uses
%	K-fold cross-validation.
%
%	X1 is the data matrix for the target class. Each row is an observation.
%	X2 is the data matrix for the non-target class. Each row is an observation.
%	K is the number to leave out in the test set for cross-validation.
%	PFA is a vector of false alarm rates.

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox


% The algorithm is to:
%
% First find pfa: (thresh)
%  Take target data, build classifier
% For each non-target class, build LR
%  Do k-leave out, build classifer on remaining, eval
%  probab
%
% Then find pcc:
%  Take all non-targets, build classifier
%  Loop over target data, leave k-out, build classifier
%    on remaining points
%  Eval probability of k points,
%  LR=p(x|target)/p(x|non-target)
%  order these, % above threshold are correctly classif
%
% class 1 is the target (x1), x2 is the non-target
% k is the number to leave out for cross-validation

[number_of_samples_target_class,p] = size(Z_input_target_class);
[number_of_samples_non_target_class,p] = size(Z_input_non_target_class);

if mod(number_of_samples_target_class,number_of_samples_for_cross_validation) ~= 0 || ...
        mod(number_of_samples_non_target_class,number_of_samples_for_cross_validation) ~= 0
    error('The sample sizes for each class must be n = r*k, for interger r.')
    break
end

likelihood_ratios_vec_for_target_samples = zeros(1,number_of_samples_target_class);
likelihood_ratios_vec_for_non_target_samples = zeros(1,number_of_samples_non_target_class);
correct_classifications_vec = zeros(size(false_alarm_probabilities_vec));

%first find the threshold corresponding to each false alarm rate
%build classifier using target data:
mu1 = mean(Z_input_target_class);
var1 = cov(Z_input_target_class);

%Do cross-validation on non-target class:
for cross_validation_counter = 1:number_of_samples_non_target_class/number_of_samples_for_cross_validation
    indices_for_test = (1+(cross_validation_counter-1)*number_of_samples_for_cross_validation) : ...
        cross_validation_counter*number_of_samples_for_cross_validation;
    data_test = Z_input_non_target_class(indices_for_test,:);
    data_left_after_taking_out_tests = Z_input_non_target_class;
    data_left_after_taking_out_tests(indices_for_test,:) = [];
    mu2 = mean(data_left_after_taking_out_tests);
    var2 = cov(data_left_after_taking_out_tests);
    likelihood_ratios_vec_for_non_target_samples(indices_for_test) = ...
           statistics_get_multivariate_normal_density_at_points(data_test,mu1,var1) ...
        ./ statistics_get_multivariate_normal_density_at_points(data_test,mu2,var2);
end
%sort the likehood ratios for the non-target class:
likelihood_ratios_vec_for_non_target_samples = sort(likelihood_ratios_vec_for_non_target_samples);
%Get the thresholds:
thresh = zeros(size(false_alarm_probabilities_vec));
for false_alarm_probability_counter = 1:length(false_alarm_probabilities_vec)
    %CHANGE THIS TO GETTING THE SAMPLE QUANTILE!!!!
    threshold_index = floor( (1-false_alarm_probabilities_vec(false_alarm_probability_counter))*number_of_samples_non_target_class );
    thresh(false_alarm_probability_counter) = likelihood_ratios_vec_for_non_target_samples(threshold_index);
end
%Now find the probability of correctly classifying targets:
mu2 = mean(Z_input_non_target_class);
var2 = cov(Z_input_non_target_class);
%Do cross-validation on target class 0:
for cross_validation_counter = 1:number_of_samples_target_class/number_of_samples_for_cross_validation
    indices_for_test = (1+(cross_validation_counter-1)*number_of_samples_for_cross_validation):number_of_samples_for_cross_validation*cross_validation_counter;
    data_test = Z_input_target_class(indices_for_test,:);
    data_left_after_taking_out_tests = Z_input_target_class;
    data_left_after_taking_out_tests(indices_for_test,:) = [];
    mu1 = mean(data_left_after_taking_out_tests);
    var1 = cov(data_left_after_taking_out_tests);
    likelihood_ratios_vec_for_target_samples(indices_for_test) = ...
                        statistics_get_multivariate_normal_density_at_points(data_test,mu1,var1) ...
                     ./ statistics_get_multivariate_normal_density_at_points(data_test,mu2,var2);
end
%find the actual pcc:
for cross_validation_counter=1:length(false_alarm_probabilities_vec)
    correct_classifications_vec(cross_validation_counter) = ...
                length(find(likelihood_ratios_vec_for_target_samples >= thresh(cross_validation_counter)));
end
correct_classifications_vec = correct_classifications_vec/number_of_samples_target_class;
