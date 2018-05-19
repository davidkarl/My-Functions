% function im = impure(pclass)
%
%  It uses the Gini index.

function im = statistics_get_impurity_gini_index(pclass)

% page 103 - CART book
%i(T) = sum_i!=j(p(omega_i|t)*p(omega_j|t))
im = 1 - sum(pclass.^2);
