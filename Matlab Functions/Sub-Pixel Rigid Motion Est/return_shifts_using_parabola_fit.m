function [shiftx,shifty] = return_shifts_using_parabola_fit(cross_corr,spacing)

%compute cross correlation:
% cross_corr=corr2_ft(mat1,mat2,1);
N=size(cross_corr,1);
if mod(N,2)==0
   midway_point=N/2; 
else
%    midway_point = (N-1)/2;
   midway_point = (N+1)/2;   
end
x=[-1,0,1];


%get fitting points for x and fit the polynomial:
% for k=1:3
%    fitting_points_x(k)=cross_corr(midway_point, midway_point+ceil(x(k))); 
%    fitting_points_y(k)=cross_corr(midway_point+ceil(x(k)) , midway_point);
% end
fitting_points_x = cross_corr(midway_point, midway_point + x);
fitting_points_y = cross_corr(midway_point + x, midway_point);

y1=fitting_points_x(1);
y2=fitting_points_x(2);
y3=fitting_points_x(3);
shiftx = (y1-y3)/(2*(y1+y3-2*y2));
y1=fitting_points_y(1);
y2=fitting_points_y(2);
y3=fitting_points_y(3);
shifty = (y1-y3)/(2*(y1+y3-2*y2));

% fitting_pixels_x = x;
% fitting_pixels_y = x;
% Px = polyfit(fitting_pixels_x',fitting_points_x',2);
% Py = polyfit(fitting_pixels_y',fitting_points_y',2);
% shiftx = -Px(2)/2/Px(1);
% shifty = -Py(2)/2/Py(1);
% if mod(N,2)==0
% shiftx = shiftx-1;
% shifty = shifty-1;
% end

shiftx=shiftx*spacing;
shifty=shifty*spacing;



