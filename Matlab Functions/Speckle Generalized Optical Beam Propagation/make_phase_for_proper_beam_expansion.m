function [proper_expansion_phase] = make_phase_for_proper_beam_expansion(expansion_factor,z,lambda,spacing,N,initial_distance_x,initial_distance_y,final_distance_x,final_distance_y)
%INITIAL AND FINAL DISTANCES ARE GIVEN AS DISTANCE FROM CENTER,
%SUCH THAT INITIAL_dISTANCE_X, FOR INSTANCE, IS 1/2*TOTAL_DISTANCE_BETWEEN_BEAMS_X

x=[-N/2:N/2-1]*spacing;
[X,Y]=meshgrid(x);
k=2*pi/lambda;
R1=z/(expansion_factor-1);
R2=z+R1;
% R=R1*(R2/z);
R=R1+1;
expansion_phase = exp(1i*k/(2*R)*((X-initial_distance_x).^2+(Y-initial_distance_y).^2));

k=2*pi/lambda;
alpha_angle_x = atan((final_distance_x-initial_distance_x)/z);
alpha_angle_y = atan((final_distance_y-initial_distance_y)/z);
tilt_phase = exp(1i*k*(final_distance_x-initial_distance_x)/z*X + 1i*k*(final_distance_y-initial_distance_y)/z*Y);

proper_expansion_phase = tilt_phase.*expansion_phase;
  