function r = statistics_get_correlation_coefficient_between_two_sets(x,y)
% CSCORR Correlation Coefficient
%   
%   COR = CSCORR(X,Y) Returns the single correlation coefficient
%   between vectors X and Y.

%   W. L. and A. R. Martinez, 12/14/07
%   Computational Statistics Toolbox, V2

data = [x(:), y(:)];
tmp = corrcoef(data);
r = tmp(1,2);