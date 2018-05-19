function mr = statistics_get_rth_centeral_moment(input_vec,r_centeral_moment)
% CSMOMENTC Sample r-th central moment.
%
%   MR = CSMOMENTC(X,R) Returns the sample R-th 
%   central moment for a given sample X.
%
%   See also CSMOMENT

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox 


n=length(input_vec);
mu=mean(input_vec);
mr = (1/n)*sum((input_vec-mu).^r_centeral_moment);
