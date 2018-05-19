function [total_beam,beam_one,beam_two,X,Y]=make_gaussian_beams_profile(w0,angle_from_x_axis,total_distance,N,spacing)

       x=[-N/2:1:N/2-1]*spacing;
       D1=spacing*N;
       [X,Y]=meshgrid(x);
       distance_x = total_distance*cos(angle_from_x_axis);
       distance_y = total_distance*sin(angle_from_x_axis);
       beam_one=exp(-((X-distance_x/2).^2+(Y-distance_y/2).^2)/w0^2);
       beam_two = exp(-((X+distance_x/2).^2+(Y+distance_y/2).^2)/w0^2);
       total_beam=beam_one+beam_two;
       total_beam = total_beam/sqrt(sum(sum(abs(total_beam).^2)));
end

    
     