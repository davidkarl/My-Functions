function [shiftx,shifty,max_x,max_y] = return_shifts_using_polynomial_fit(cross_corr,spacing,polynom_order,number_of_fitting_points,flag)

%compute cross correlation:
% cross_corr=corr2_ft(mat1,mat2,1);
N=size(cross_corr,1);

%get relevant x-axis and y-axis:
[cross_section_vec_x,cross_section_x,max_row,max_col] = get_cross_section_at_maximum(cross_corr,spacing,1);
[cross_section_vec_y,cross_section_y,max_row,max_col] = get_cross_section_at_maximum(cross_corr,spacing,0);

%define pixels for the fit
first_pixel_x=max_col-floor(number_of_fitting_points/2);
last_pixel_x=max_col+(number_of_fitting_points-floor(number_of_fitting_points/2))-1;
first_pixel_y=max_row-floor(number_of_fitting_points/2);
last_pixel_y=max_row+(number_of_fitting_points-floor(number_of_fitting_points/2))-1;

%get fitting points for x and fit the polynomial:
fitting_points_x = cross_section_x(first_pixel_x:last_pixel_x);
fitting_pixels_x = [first_pixel_x:last_pixel_x];
Px = polyfit(fitting_pixels_x,fitting_points_x,polynom_order);

%get fitting points for y and fit the polynomial:
fitting_points_y = cross_section_y(first_pixel_y:last_pixel_y);
fitting_pixels_y = [first_pixel_y:last_pixel_y];
Py = polyfit(fitting_pixels_y,fitting_points_y',polynom_order);

% flag=1 for fft cross correlation
% flag=2 for regular cross correlation

%FIND MAX OF THE POLYNOMIALS:
%first, find the derivative of the polynomials:
derivative_x=polyder(Px);
derivative_y=polyder(Py);
%now find the zeros of the derivative:
roots_x=roots(derivative_x);
roots_y=roots(derivative_y);
%now only use the roots that are between the points we choose to fit by:
shiftx=roots_x(roots_x>=first_pixel_x & roots_x<=last_pixel_x);
shifty=roots_y(roots_y>=first_pixel_y & roots_y<=last_pixel_y);
%now find the maximum out of the roots which are relevant:
max_x=max(polyval(Px,roots_x));
max_y=max(polyval(Py,roots_y));
%now convert the calculated shifts into "real world" displacements:
if mod(N,2)==0
if flag==1
shiftx=-(shiftx-N/2-1)*spacing;
shifty=-(shifty-N/2-1)*spacing;
elseif flag==2
shiftx=-(shiftx-N/2)*spacing;
shifty=-(shifty-N/2)*spacing;
end
else
shiftx=-(shiftx-ceil(N/2))*spacing;
shifty=-(shifty-ceil(N/2))*spacing;    
end

% %plot detailed fitted polynomial
% detailed_number=200;
% fitted_graph_x=zeros(1,detailed_number); %initialize detailed fitted graph for x
% fitted_graph_y=zeros(1,detailed_number); %initialize detailed fitted graph for y
% detailed_fitting_points_x = linspace(first_pixel_x,last_pixel_x,detailed_number);
% detailed_fitting_points_y = linspace(first_pixel_y,last_pixel_y,detailed_number);
% %fit detailed graph just for visual
% fitted_graph_x=polyval(Px,detailed_fitting_points_x);
% fitted_graph_y=polyval(Py,detailed_fitting_points_y);
% subplot(1,2,1)
% plot(fitting_pixels_x,fitting_points_x,'b*',detailed_fitting_points_x,fitted_graph_x,'m--');
% subplot(1,2,2)
% plot(fitting_pixels_y,fitting_points_y,'b*',detailed_fitting_points_y,fitted_graph_y,'m--');






    

    