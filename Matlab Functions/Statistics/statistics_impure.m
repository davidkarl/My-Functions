% function im = impure(pclass)
%
%  It uses the Gini index.

function im = statistics_impure(pclass)

% page 103 - CART book
im = 1 - sum(pclass.^2);
