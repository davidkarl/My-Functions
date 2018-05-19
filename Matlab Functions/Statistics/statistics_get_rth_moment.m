function mr = statistics_get_rth_moment(input_vec,r_moment)
% CSMOMENT r-th sample moment.
%
%   MR = CSMOMENT(X,R) Returns the R-th sample moment
%   using the sample given in X.
%
%   See also CSMOMENTC

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox 

n=length(input_vec);
mu=mean(input_vec);
mr = (1/n)*sum(input_vec.^r_moment);
