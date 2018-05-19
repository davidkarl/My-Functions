function [shiftx,shifty] = return_shifts_using_paraboloid_fit(cross_correlation,spacing)

N=size(cross_correlation,1);

% if mod(N,2)==0
%    midway_point = N/2; 
%    x=[0,0,0,1,1,1,2,2,2];
%    y=[0,1,2,0,1,2,0,1,2];
% else
%    midway_point = (N-1)/2;
%    x=[-1,-1,-1,0,0,0,1,1,1];
%    y=[-1,0,1,-1,0,1,-1,0,1];
% end

[max1,loc1] = max(cross_correlation);
[max2,loc2] = max(max1);
rloc=loc1(loc2);cloc=loc2;
x=[cloc-1,cloc-1,cloc-1,cloc,cloc,cloc,cloc+1,cloc+1,cloc+1];
y=[rloc-1,rloc,rloc+1,rloc-1,rloc,rloc+1,rloc-1,rloc,rloc+1];

for k=1:length(x)
   cross_correlation_samples(k) = cross_correlation(x(k),y(k)); 
end
[coeffs] = fit_polynom_surface( x', y', cross_correlation_samples', 2 );


% y_c = -(B*E - 2*C*D)/(E^2 - 4*C*F)
% x_c = (2*B*F - D*E)/(E^2 - 4*C*F)
shifty = (-(coeffs(2)*coeffs(5)-2*coeffs(3)*coeffs(4))/(coeffs(5)^2-4*coeffs(3)*coeffs(6)));
shiftx = ((2*coeffs(2)*coeffs(6)-coeffs(4)*coeffs(5))/(coeffs(5)^2-4*coeffs(3)*coeffs(6)));
if mod(N,2)==0
shiftx=shiftx-N/2-1;
shifty=shifty-N/2-1;
else
shiftx=shiftx-ceil(N)/2;
shifty=shifty-ceil(N)/2;    
end
% if mod(N,2)==0
%    shiftx=shiftx-1;
%    shifty=shifty-1;
% end
shiftx=shiftx*spacing;
shifty=shifty*spacing;

