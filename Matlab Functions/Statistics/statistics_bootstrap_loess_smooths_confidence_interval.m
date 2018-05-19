function [low, up] = statistics_bootstrap_loess_smooths_confidence_interval(...
                            x_input_points,...
                            y_input_points,...
                            number_of_bootstrap_replicates,...
                            alpha_confidence_interval,...
                            alpha_loess_parameter,...
                            deg_loess_parameter)

%   CSBOOTBAND      Bootstrap confidence bands for loess smooths
%
%   [LOW, UP] = CSBOOTBAND(X,Y,B,CONF,ALPHA,DEGREE) 
%   returns the CONF*100% confidence bands for loess smooth at the  
%   predictor values X and given response values Y.
% 
%   The number of bootstrap replicates is given by B.
%
%   The input arguments ALPHA & DEGREE are the parameters for the LOESS
%   function.
%
%   The output consists of two vectors - the lower and upper pointwise
%   confidence bands.
%
%   EXAMPLE:
%
%   load environmental
%   % Sort the x values.
%   [x,ind] = sort(Temperature);
%   y = Ozone(ind);
%   n = length(x);
%   B = 200;
%   yhat = csloess(x,y,x,2/3,1);
%   [low, up] = csbootband(x,y,B,0.95,2/3,1);
%   % Plot the smooth and the confidence bands
%   plot(x,y,'.',x,low,':',x,up,':',x,yhat)
%
%   See also CSLOESSENV,  CSLOESS

%   W. L. and A. R. Martinez, 12/14/07
%   Computational Statistics Toolbox, V 2 

n = length(x_input_points);
% Create a matrix to keep the bootstrap
% regression fits. Each row will be a fit 
% evaluated at x_i. 
Bmat = zeros(number_of_bootstrap_replicates,n);
h = waitbar(0,'Please wait creating the bands...');
for i = 1:number_of_bootstrap_replicates
    % Get the bootstrap sample.
    indB = unidrnd(n,1,n); % Statistics Toolbox
    yhatB = csloess(x_input_points(indB),y_input_points(indB),x_input_points,alpha_loess_parameter,deg_loess_parameter);
    Bmat(i,:) = yhatB;
    waitbar(i/number_of_bootstrap_replicates,h)
end
delete(h)
% Sort the values in each column.
Bmats = sort(Bmat);
a = 1 - alpha_confidence_interval;
tl = a/2;
tu = 1- a/2;
low = Bmats(round(tl*number_of_bootstrap_replicates),:);
up = Bmats(round(tu*number_of_bootstrap_replicates),:);





