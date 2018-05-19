function subband_angle = get_PDFB_subband_angle(subband_index, number_of_levels)
% PDFB_ANG Determine the dfb subband that the angle fall into
%       ang = pdfb_ang(insub, nlev)
% 
% Returning value ang is from 3pi*4 to -pi/4 to make the range of value
% change continuosly from band 1 to 2^nlev
% Input:
%   insub:  index of the subband from 1 to 2^nlev
%   nlev:   2^nlev is the number of direction band
%
% Output:
%   ang:	the angle of direction of the band in radian
%
% Note:
% Note: An important thing to remember is that the ang is limited to -pi/2
% to pi/2, but the actual angle of the complex filter is from -pi to pi.
% This is because the complex filter contained a imaginary anti-symmetric
% component. We consider the direction of the impulse responses is that if
% we go in the positive direction, the right hand side will corresponds to
% the positive (larger than zero) of an antisymmetric wave (forexample,
% sine function)
%                               ^ pi/2
%                        \      |      /band 2^(N-1)  
%                          \    |    /    
%                            \  |  / 
%                              \|/   +  band 2^(N-1)+2^(N-2)
%                               |----------------->
%                               |\   -
%                               |  \  
%                               |    \  band 2^N - 1 
%                               |   b0 \ 
%                               |-pi/2
% See also: ANG_PDFB

subband_index = 1;
number_of_levels = 3;

%number of all sb:
total_number_of_subbands = 2^(number_of_levels-2);

%artang value of the smallest angle:
atan_smallest_angle = 1/(total_number_of_subbands*2);
%artang step from adjacent subband:
atan_angular_step = 1/(total_number_of_subbands);
%artang vec:
atan_vec = atan_smallest_angle:atan_angular_step:atan_smallest_angle ...
                                                            + atan_angular_step*(total_number_of_subbands-1);
%GET ALL THE SUBBAND ANGLES - correspoding angle value in radian:
subbands_angle_vec = atan(atan_vec);
subbands_angle_vec_reversed = fliplr(subbands_angle_vec);

%angle of the four group of direction subbands:
aq1 = subbands_angle_vec_reversed + pi/2;
aq2 = -fliplr(subbands_angle_vec_reversed - pi/2);
aq3 = subbands_angle_vec_reversed;
aq4 = -fliplr(aq3);
aq = [aq1, aq2, aq3, aq4];
bla = aq/pi; 
subband_angle = aq(subband_index);
