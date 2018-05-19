function projection_pursuit_index = statistics_chi_square_projection_pursuit_index(...
                                Z_input_data_mat,a_plane,b_plane,number_of_samples,ck)
% CSPPIND Chi-square projection pursuit index.
%   
%   PPI = CSPPIND(Z,ALPHA,BETA,N,CK)
%   This finds the value of the projection pursuit index
%   for a plane spanned by the column vectors ALPHA and
%   BETA. The vector CK contains the bivariate standard
%   normal probabilities for radial boxes. CK is usually
%   found in the function CSPPEDA. The matrix Z is the
%   sphered or standardized version of the data.
%
%   See also CSPPEDA, CSPPSTRTREM

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox 

probability_of_bivariate_standard_normal_over_each_radial_box

rotation_angles_vec_length = 9;
Z_input_data_mat_rotated = zeros(number_of_samples,2);
projection_pursuit_index = 0;
pk = zeros(1,48);
eta_angles = pi*(0:(rotation_angles_vec_length-1))/36;
delta_angle = 45*pi/180;
delta_radiuses = sqrt(2*log(6))/5;
angles_vec = 0:delta_angle:(2*pi);
radiuses_vec = 0:delta_radiuses:5*delta_radiuses;
radius_vec_length = length(radiuses_vec);
angles_vec_length = length(angles_vec);

for rotation_angle_index = 1:9
   %find rotated plane:
   aj = a_plane*cos(eta_angles(rotation_angle_index)) - b_plane*sin(eta_angles(rotation_angle_index));
   bj = a_plane*sin(eta_angles(rotation_angle_index)) + b_plane*cos(eta_angles(rotation_angle_index));
   
   %project data onto this plane:
   Z_input_data_mat_rotated(:,1) = Z_input_data_mat*aj;
   Z_input_data_mat_rotated(:,2) = Z_input_data_mat*bj;
   
   %convert to polar coordinates:
   [th,r] = cart2pol(Z_input_data_mat_rotated(:,1),Z_input_data_mat_rotated(:,2));
   
   %find all of the angles that are negative:
   indices_within_slice = find(th<0);
   th(indices_within_slice) = th(indices_within_slice) + 2*pi;
   
   %find # points in each box:
   for radius_counter = 1:(radius_vec_length-1)	% loop over each ring
      for angle_counter = 1:(angles_vec_length-1)	% loop over each wedge
          %Current slice index:
          slice_index = (radius_counter-1)*rotation_angles_vec_length + angle_counter;
          
          %Find input data points inside current slice in radii and angle:
          indices_within_slice = find(r>radiuses_vec(radius_counter) & r<radiuses_vec(radius_counter+1) ...
                                    & th>angles_vec(angle_counter) & th<angles_vec(angle_counter+1));
          
          pk(slice_index) = (length(indices_within_slice)/number_of_samples - ck(slice_index))^2 / ck(slice_index);
      end
   end
   %find the number in the outer line of boxes:
   for angle_counter = 1:(angles_vec_length-1)
      indices_within_slice = find(r>radiuses_vec(radius_vec_length) ...
                                & th>angles_vec(angle_counter) & th<angles_vec(angle_counter+1));
      pk(40+angle_counter) = (length(indices_within_slice)/number_of_samples - (1/48))^2 / (1/48);
   end
   
   
   projection_pursuit_index = projection_pursuit_index + sum(pk);
end


projection_pursuit_index = projection_pursuit_index/9;


