function X = statistics_generate_random_samples_from_beta_distribution(alpha_parameter,beta_parameter,number_of_samples)
% CSBETARND Generate variates from the univariate beta distribution.
%
%   Y = CSBETARND(ALPHA,BETA,N) Returns a row vector of random
%   variables from the beta distribution with parameters
%   ALPHA and BETA.
%
%   See also CSBETAP, CSBETAC

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox

if alpha_parameter == 1 && beta_parameter ==1 % then the beta is uniform
    X = rand(1,number_of_samples);
    
else  % use acceptance-rejection method
    % set up the space to store the random variables.
    X = zeros(1,number_of_samples);
    % get the constant in the beta pdf
    const = gamma(alpha_parameter+beta_parameter)/(gamma(alpha_parameter)*gamma(beta_parameter));
    % get the value of pdf at the mode
    if alpha_parameter > 1 && beta_parameter > 1
        arg1 = (alpha_parameter-1)/(alpha_parameter+beta_parameter-2);
        arg2 = (beta_parameter-1)/(beta_parameter+alpha_parameter-2);
        mode = const*arg1^(alpha_parameter-1)*arg2^(beta_parameter-1);
    else
        mode = 4;   % set peak to arbitrary value
    end
    % start generating the random variables
    i=1; 
    while i <= number_of_samples
        % this will be used to evaluate the beta pdf
        u1 = rand(1);
        u2 = rand(1);
        tmp = const*u1^(alpha_parameter-1)*(1-u1)^(beta_parameter-1);
        if mode*u2 <= tmp
            % accept that variate
            X(i)=u1;
            i=i+1;
        end
    end 
end


