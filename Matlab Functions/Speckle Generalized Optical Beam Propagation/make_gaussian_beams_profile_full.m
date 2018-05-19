function [total_beam,beam_one,beam_two,X,Y]=make_gaussian_beams_profile_full(w0,distance_x,distance_y,beam_with_changing_size,beam_size_difference_ratio,beam_with_changing_intensity,beam_intensity_difference_ratio,N,spacing)

       x=[-N/2:1:N/2-1]*spacing;
       y=x;
       [X,Y]=meshgrid(x,y);
       
       %beam waists:
       w1 = w0;
       w2 = w0*beam_size_difference_ratio;
       
       %build normalized beams:
       if beam_with_changing_size==0 %left beam changes
           beam_one=exp(-((X-distance_x/2).^2+(Y-distance_y/2).^2)/w1^2);
           beam_one = beam_one/sqrt(sum(sum(abs(beam_one).^2)));
           beam_two = exp(-((X+distance_x/2).^2+(Y+distance_y/2).^2)/w2^2);
           beam_two = beam_two/sqrt(sum(sum(abs(beam_two).^2)));
       else %right beam changes
           beam_one=exp(-((X-distance_x/2).^2+(Y-distance_y/2).^2)/w2^2);
           beam_one = beam_one/sqrt(sum(sum(abs(beam_one).^2)));
           beam_two = exp(-((X+distance_x/2).^2+(Y+distance_y/2).^2)/w1^2);
           beam_two = beam_two/sqrt(sum(sum(abs(beam_two).^2)));
       end
      
       %change beam intensity:
       if beam_with_changing_intensity==0 %left beam changes
           beam_two = beam_two*beam_intensity_difference_ratio;
       else %right beam changes
           beam_one = beam_one*beam_intensity_difference_ratio;
       end
       
       %normalize total beam:
       total_beam=beam_one+beam_two;
       total_beam = total_beam/sqrt(sum(sum(abs(total_beam).^2)));
end

    
    