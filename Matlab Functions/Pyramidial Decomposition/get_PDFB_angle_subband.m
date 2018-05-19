function subband_index = get_PDFB_angle_subband(subband_angle, number_of_levels)
% ANG_PDFB Determine the dfb subband that the angle fall into
%       insub = ang_pdfb(ang, nlev)
% 
% Input:
%   ang:	the angle of direction of the band in radian
%   nlev:   2^nlev is the number of direction band
%
% Output:
%   insub:  index of the subband 
%
% Note: An important thing to remember is that the ang is limited to -pi/2
% to pi/2, but the actual angle of the complex filter is from -pi to pi.
% This is because the complex filter contained a imaginary anti-symmetric
% component. We consider the direction of the impulse responses is that if
% we go in the positive direction, the right hand side will corresponds to
% the positive (larger than zero) of the antisymmetric wave/
%                               ^ pi/2
%                               |      /band 2^(N-1)  
%                               |    /    
%                               |  / 
%                               |/   +  band 2^(N-1)+2^(N-2)
%                               |----------------->
%                               |\   -
%                               |  \  
%                               |    \  band 2^N - 1 
%                               |   b0 \ 
%                               |-pi/2
% See also: PDFB_ANG

% nlev = 4;
% number of all sb 
total_number_of_subbands = 2^number_of_levels;
if abs(subband_angle) > pi/2
    subband_angle = mod(subband_angle+pi/2, pi) - pi/2;
end

% artang value of the smallest angle
atan_smallest_angle = 1/(total_number_of_subbands*2);
% artang step from adjacent subband
atan_angular_step = 4/(total_number_of_subbands);
% step of argtan value

if abs(subband_angle) < pi/4
    alpha = tan(subband_angle);
    subband_index = fix( (-alpha +1)/ atan_angular_step) + total_number_of_subbands/2 +1;
else
    alpha = 1/tan(subband_angle);
    subband_index = fix( (alpha +1)/ atan_angular_step) + 1;
end

