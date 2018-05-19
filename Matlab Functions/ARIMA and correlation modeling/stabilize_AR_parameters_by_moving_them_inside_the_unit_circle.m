function [AR_parameters_new] = stabilize_AR_parameters_by_moving_them_inside_the_unit_circle(AR_parameters)
% Given parameters of AR filter, stabilize them if necessary by moving them
% inside the unit circle
% Usage: anew=astab(aold)

n = length(AR_parameters) ;
AR_equation_roots = roots([1 ,-AR_parameters]) ;
max_AR_equation_root = max(abs(AR_equation_roots)) ;
closenes_to_unit_circle_allowed = 1-1e-3 ;

if max_AR_equation_root>=closenes_to_unit_circle_allowed
  
    %bring max AR root as close as defined to the unit circle:
    AR_equation_roots = closenes_to_unit_circle_allowed * (AR_equation_roots / max_AR_equation_root);
    
    %get new AR parameters using new difference equation 
    AR_parameters_new = real(poly(AR_equation_roots));
    
    %normalize their form with respect to AR_parameters_new(1) (usually=1):
    AR_parameters_new = -AR_parameters_new(2:n+1)/AR_parameters_new(1);

else   
    %parameters are stable - do nothing:
    AR_parameters_new = AR_parameters;
end
