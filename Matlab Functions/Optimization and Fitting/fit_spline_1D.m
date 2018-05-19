function [y_splined,spline_result,new_x] = fit_spline_1D(x,y,spline_order,upsample_factor)
%assuming x is constant spacing:

%1D array:
x=x(:);
new_x = interp1(x,x,linspace(x(1),x(end),length(x)*upsample_factor-(upsample_factor-1)));
y=y(:);
spline_result = spapi(spline_order,x,y);
y_splined = fnval(spline_result,new_x);
