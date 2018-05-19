function [total_beam,beam_one,beam_two,X,Y]=make_gaussian_beams_profile_with_expansion_phase(w0,distance_x,distance_y,expansion_factor,z,lambda,N,spacing)

       x=[-N/2:1:N/2-1]*spacing;
       D1=spacing*N;
       y=x;
       [X,Y]=meshgrid(x,y);
       
       k=2*pi/lambda;
       R1=z/(expansion_factor-1);
       R2=z+R1;
       R=R1+1;
       
       %truncate the expansion phase to twice the initial waist size to
       %avoid affecting the other beam when combined together at "total_beam":
       truncating_aperture_for_phase1 = ((X-distance_x/2).^2+(Y-distance_y/2).^2<=(2*w0)^2);
       truncating_aperture_for_phase2 = ((X+distance_x/2).^2+(Y+distance_y/2).^2<=(2*w0)^2);
       
       proper_phase1 = exp(1i.*truncating_aperture_for_phase1*k/(2*R).*((X-distance_x/2).^2+(Y-distance_y/2).^2));
       proper_phase2 = exp(1i.*truncating_aperture_for_phase2*k/(2*R).*((X+distance_x/2).^2+(Y+distance_y/2).^2));
       
       beam_one = exp(-((X-distance_x/2).^2+(Y-distance_y/2).^2)/w0^2).*proper_phase1;
       beam_two = exp(-((X+distance_x/2).^2+(Y+distance_y/2).^2)/w0^2).*proper_phase2;
       total_beam=beam_one+beam_two;
       total_beam = total_beam/sqrt(sum(sum(abs(total_beam).^2)));
    end

    
    