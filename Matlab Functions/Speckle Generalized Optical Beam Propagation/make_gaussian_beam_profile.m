function [total_beam,X,Y]=make_gaussian_beam_profile(w0,total_distance,angle_from_x_axis,N,spacing)
       angle_from_x_axis_corrected = pi/180*angle_from_x_axis;
       distance_x = total_distance*cos(angle_from_x_axis_corrected);
       distance_y = total_distance*sin(angle_from_x_axis_corrected);
       x=[-N/2:1:N/2-1]*spacing;
       [X,Y]=meshgrid(x);
       total_beam=exp(-((X-distance_x).^2+(Y-distance_y).^2)/w0^2);
       total_beam = total_beam/sqrt(sum(sum(abs(total_beam).^2)));
end
 