function [coeffs_mat] = fit_2D_polynomial(x,y,z,deg_vec,flag_fit_constant,sigma_correction,max_number_of_iterations,varargin)
%(1). deg_vec = [deg_x,deg_y,deg_xy]
%(2). if flag_up_to_or_only_top_degree==1 then use all degrees up to the
%     relevant deg_vec component, otherwise only fit the max degree.
%(3). be careful when using many iterations or sigma_correction which is
%     too small...it might be good for big outlier but not for much more
%(4). if flag_specify_max_deg_or_specific_coefficients==2 then specify
%     orders by [degx1,degy1;degx2,degy2....]

%Decide, according to deg_vec format, what it means:
if size(deg_vec,2) == 3
   flag_specify_max_deg_or_specific_coefficients=1;
   %I DON'T ALLOW THE SECOND OPTION FOR NOW, I HAVEN'T COMPLETED IT AND ITS
   %TOO UNIMPORTANT
else
   error('deg_vec should be made out of deg(x^a),deg(y^b),deg(x^a*y^b)');
end

%Check input:
if nargin<=4
    %Default:
    flag_fit_constant = 1;
    sigma_correction = 3;
    max_number_of_iterations = 1;
elseif nargin<=5
   %Default:
   sigma_correction = 3;
   max_number_of_iterations = 1;
elseif nargin<=6
   %Default:
   max_number_of_iterations = 1;
end


%Make x,y,z column vectors:
x = x(:);
y = y(:);
z = z(:);

%Initialize transfer of coeffs vec into a coeffs_mat:
max_deg = max(max(deg_vec(1),deg_vec(2)),deg_vec(3)-1);
if flag_specify_max_deg_or_specific_coefficients==1
    coeffs_mat = zeros(max_deg,max_deg);
elseif flag_specify_max_deg_or_specific_coefficients==2
    coeffs_mat = zeros(max(deg_vec(:))+1);
end

%Get number of values:
number_of_values = length(x);

%Scale to prevent precision problems
scalex = 1.0/max(abs(x));
scaley = 1.0/max(abs(y));
scalez = 1.0/max(abs(z));
xs = x .* scalex;
ys = y .* scaley;
zs = z .* scalez;

%Build design matrix:
%Constant term:
%Get number of coefficients (const + x_degs + y_degs + xy_degs_combinations):
number_of_coefficients = 1 + deg_vec(1) + deg_vec(2) + deg_vec(3)*(deg_vec(3)+1)/2; 
if (flag_fit_constant==1) 
    H = ones(length(x),1);
else
    H = []; 
end 
%Only X-degrees (without constant term):
for current_deg_x=1:deg_vec(1)
    H = [H, xs.^current_deg_x];
end
%Only Y-degrees (without constant term):
for current_deg_y=1:deg_vec(2)
    H = [H, ys.^current_deg_y];
end
%Combined x^i*y^j:
for current_deg_x = 1:deg_vec(3)
    for current_deg_y = 1:(deg_vec(3)-current_deg_x)
        H = [H, (xs.^current_deg_x) .* (ys.^current_deg_y)];
    end
end


%Solve LSQ problem:
iterations_counter = 1;
is_observation_within_uncertainty_logical_vec = ones(length(x),1);
current_number_of_values = number_of_values;
while (iterations_counter<=max_number_of_iterations)
    
    %Get indices which conform to fit such that their observation is within sigma*std_fit:
    iterations_counter = iterations_counter + 1;
    is_observation_within_uncertainty_indices = find(is_observation_within_uncertainty_logical_vec);
    
    %Perform SVD
    [u, s, v] = svd(H(is_observation_within_uncertainty_indices,:));
    
    %Pseudo-Inverse of diagonal matrix s
    sigma = eps^(1/max_deg); % minimum value considered non-zero
    qqs = diag(s);
    qqs(abs(qqs)>=sigma) = 1./qqs(abs(qqs)>=sigma);
    qqs(abs(qqs)<sigma)=0;
    qqs = diag(qqs);
    if current_number_of_values > number_of_coefficients
        qqs(current_number_of_values,1)=0; % add empty rows
    end
    
    %Calculate  coeffs_vec solution:
    coeffs_vec = v*qqs'*u'*zs(is_observation_within_uncertainty_indices);
    
    %Calculate residuals and check where observation is within model wanted uncertainty:
    residual_vec = zs(is_observation_within_uncertainty_indices) - H(is_observation_within_uncertainty_indices,:)*coeffs_vec;
    residual_vec_std = std(residual_vec);
    is_observation_within_uncertainty_logical_vec = (residual_vec<sigma_correction*residual_vec_std) & residual_vec>(-sigma_correction*residual_vec_std);
    current_number_of_values = sum(is_observation_within_uncertainty_logical_vec);
    
    %Rescale the coefficients so they are correct for the unscaled data and
    %insert values into properly created coeffs_mat
    counter = 1;
    %Constant term:
    if flag_fit_constant==1
        coeffs_vec(1) = coeffs_vec(1) * scalex^0 * scaley^0 / scalez;
        coeffs_mat(1,1) = coeffs_vec(1);
        counter = counter+1;
    end
    %Only X-degrees (without constant term):
    for current_deg_x=1:deg_vec(1)
        coeffs_vec(counter) = coeffs_vec(counter) * scalex^current_deg_x / scalez;
        coeffs_mat(current_deg_x+1,1) = coeffs_vec(counter);
        counter = counter+1;
    end
    %Only Y-degrees (without constant term):
    for current_deg_y=1:deg_vec(2)
        coeffs_vec(counter) = coeffs_vec(counter) * scaley^current_deg_y / scalez;
        coeffs_mat(1,current_deg_y+1) = coeffs_vec(counter);
        counter = counter+1;
    end
    %Combined x^i*y^j:
    for current_deg_x = 1:deg_vec(3)
        for current_deg_y = 1:(deg_vec(3)-current_deg_x)
            coeffs_vec(counter) = coeffs_vec(counter) * scalex^current_deg_x * scaley^current_deg_y / scalez;
            coeffs_mat(current_deg_x+1,current_deg_y+1) = coeffs_vec(counter);
            counter=counter+1;
        end
    end
   
end



