function [beta,parameter_list] = fit_2D_gaussian(x,y,z,flag_lsq_or_and_nonlinear,flag_same_sigma,flag_fit_correlated,flag_fit_x0,flag_fit_y0,flag_fit_background,options,vargin)
%flag_lsq_or_nonlinear = 1 -> only lsq
%flag_lsq_or_nonlinear = 2 -> only nonlinear (With default initial guess)
%flag_lsq_or_nonlinear = 3 -> nonlinear with initial guess based on lsq

% %Test initialization:
% X=-5:5;
% [x,y]=meshgrid(X);
% z=3.5*exp(-1/2*(x.^2+y.^2));
% flag_same_sigma=1;
% flag_fit_correlated=1;
% flag_fit_x0=1;
% flag_fit_y0=1;
% flag_fit_background=1;
% flag_lsq_or_and_nonlinear = 1;
% options = [];

%Default values:
flag_lsq_or_nonlinear_default = 1;
flag_same_sigma_default = 0;
flag_correlated_default = 1;
flag_fit_x0_default = 1;
flag_fit_y0_default = 1;
flag_fit_background_default = 1;
if nargin<4
    flag_lsq_or_and_nonlinear = flag_lsq_or_nonlinear_default;
    flag_same_sigma = flag_same_sigma_default;
    flag_fit_correlated = flag_correlated_default;
    flag_fit_x0 = flag_fit_x0_default;
    flag_fit_y0 = flag_fit_y0_default;
    flag_fit_background = flag_fit_background_default;
    options = [];
elseif nargin<5
    flag_same_sigma = flag_same_sigma_default;
    flag_fit_correlated = flag_correlated_default;
    flag_fit_x0 = flag_fit_x0_default;
    flag_fit_y0 = flag_fit_y0_default;
    flag_fit_background = flag_fit_background_default;
    options = [];
elseif nargin<6
    flag_fit_correlated = flag_correlated_default;
    flag_fit_x0 = flag_fit_x0_default;
    flag_fit_y0 = flag_fit_y0_default;
    flag_fit_background = flag_fit_background_default;
    options = [];
elseif nargin<7
    flag_fit_x0 = flag_fit_x0_default;
    flag_fit_y0 = flag_fit_y0_default;
    flag_fit_background = flag_fit_background_default;
    options = [];
elseif nargin<8
    flag_fit_y0 = flag_fit_y0_default;
    flag_fit_background = flag_fit_background_default;
    options = [];
elseif nargin<9
    flag_fit_background = flag_fit_background_default;
    options = [];
elseif nargin<10
    options = [];
end
  
%hold original z for debugging:
original_z = z;
original_x = x;
original_y = y;

%turn x and y into row vectors:
x=x(:); 
y=y(:); 
z=z(:);

%Do lsq fitting if wanted:
if flag_lsq_or_and_nonlinear==1 || flag_lsq_or_and_nonlinear == 3
   
   %offset values so that there would be no negative values: 
   original_offset = min(z(:));
   if flag_fit_background==1
       z = z - original_offset; 
   end
   
   %chop off entries where z=0 to prevent infinities due to numerical
   %inaccuracies when the gaussian values are so small that they get
   %swalloed in the DC: 
   okay_indices_in_z = find((z>0)); 
   z = z(okay_indices_in_z);
   x = x(okay_indices_in_z);
   y = y(okay_indices_in_z);
   
   %Scale to prevent precision problems
   scale_x = 1.0/max(abs(x));
   scale_y = 1.0/max(abs(y));
   scale_z = 1.0/max(abs(z));
   xs = x .* scale_x;
   ys = y .* scale_y;
   zs = z .* scale_z;
   
   %take log(z) to be able to use lsq:
   zs = log(zs);
   
   counter=1;
   %constant term in exponent ~ Amplitude of gaussian and gaussian center:
   H = ones(length(x),1);
   degree_vec(counter,:) = [0,0];
   counter=counter+1;
   %parabolic termms:
   if flag_same_sigma==0
       H = [H, xs.^2,ys.^2];
       degree_vec(counter,:) = [2,0];
       degree_vec(counter+1,:) = [0,2];
       counter=counter+2;
   else 
       H = [H, xs.^2+ys.^2];
       degree_vec(counter,:) = [2,2];
       counter=counter+1; 
   end  
   %mixed terms:
   if flag_fit_correlated==1
      H = [H, xs.*ys];
      degree_vec(counter,:) = [1,1];
      counter=counter+1;
   else
      x0y0 = 0; 
   end 
   %linear terms:
   if flag_fit_x0==1
      H = [H, xs]; 
      degree_vec(counter,:) = [1,0];
      counter=counter+1;
   else
      x0 = 0; 
      Ax = 0;
   end
   if flag_fit_y0==1
      H = [H, ys]; 
      degree_vec(counter,:) = [0,1];
      counter=counter+1;
   else
      y0 = 0;
      Ay = 0;
   end
   
   
   %Perform SVD
   [u, s, v] = svd(H);

   %Pseudo-Inverse of diagonal matrix s
   number_of_values = length(x);
   number_of_coefficients = counter-1;
   sigma = eps^(1/2); % minimum value considered non-zero
   qqs = diag(s);
   qqs(abs(qqs)>=sigma) = 1./qqs(abs(qqs)>=sigma);
   qqs(abs(qqs)<sigma)=0;
   qqs = diag(qqs);
   if number_of_values > number_of_coefficients
       qqs(number_of_values,1)=0; % add empty rows
   end

   %Calculate  coeffs_vec solution:
   coeffs_vec = v*qqs'*u'*zs;
   
   %Rescale coefficients:
   for k=1:counter-1
      coeffs_vec(k) = coeffs_vec(k) * scale_x^(degree_vec(k,1)) * scale_y^(degree_vec(k,2)); 
   end
   
   %Use coefficients to create new coefficients for nonlinear fit:
   if flag_same_sigma==0
       a = coeffs_vec(2); %a*x^2
       b = coeffs_vec(3); %b*y^2
       counter = 3;
   else 
       a = coeffs_vec(2);
       b = coeffs_vec(2);
       counter = 2;
   end
   if flag_fit_correlated==1  
      %Fit correlated term:
      c = coeffs_vec(counter+1); %c*x*y
      if flag_fit_x0==1 && flag_fit_y0==0
         Ax = coeffs_vec(counter+2); %Ax*x
         Ay = 0;
         x0 = (Ay/2/b-Ax/c)/(2*a/c-c/2/b);
         y0 = 0;
      end
      if flag_fit_x0==0 && flag_fit_y0==1
         Ay = coeffs_vec(counter+2); %Ay*y
         y0 = (Ay/c-Ax/2/a)/(c/2/a-2*b/c);
         x0 = 0;
         Ax = 0;
      end
      if flag_fit_x0==1 && flag_fit_y0==1
         Ax = coeffs_vec(counter+2); %Ax*x
         Ay = coeffs_vec(counter+3); %Ay*y
         x0 = (Ay/2/b-Ax/c)/(2*a/c-c/2/b);
         y0 = (Ay/c-Ax/2/a)/(c/2/a-2*b/c);
      else
         x0=0;
         y0=0;
      end
   else
      %Don't fit correlated term:
      c = 0; 
      if flag_fit_x0==1 && flag_fit_y0==0
         Ax = coeffs_vec(counter+1); %Ax*x
         Ay = 0;
         x0 = -Ax/2/a;
         y0 = 0;
      end
      if flag_fit_x0==0 && flag_fit_y0==1
         Ay = coeffs_vec(counter+1); %Ay*y
         y0 = -Ay/2/b;
         x0 = 0;
         Ax = 0;
      end
      if flag_fit_x0==1 && flag_fit_y0==1
         Ax = coeffs_vec(counter+1); %Ax*x
         Ay = coeffs_vec(counter+2); %Ay*y
         x0 = -Ax/2/a;
         y0 = -Ay/2/b;
      else
         x0=0; 
         y0=0;
      end
   end
   
   %Use results to guess Amplitude coeffs:
   N = exp(coeffs_vec(1) - a*x0^2-b*y0^2-c*x0*y0)/scale_z;
   
   %turn initial results into more understandable parameters.
   %a=1/(sigma^2)
   sigma_x = sqrt(1/a);
   sigma_y = sqrt(1/b);
   rho = 1-c;
   
   
   %Build initial parameter guess vec and final parameter list:
   counter=1; 
   guess_vec = [];
   if flag_fit_background
      B = original_offset;
   else
      B = 0; 
   end
   if flag_fit_background==1
       guess_vec = [guess_vec, B];
       parameter_list{counter} = 'Background';
       counter=counter+1;
   end
   guess_vec = [guess_vec, N];
   parameter_list{counter} = 'Normalization';
   counter=counter+1;
   guess_vec = [guess_vec, a];
   parameter_list{counter} = 'Ax';
   counter=counter+1;
   if flag_fit_x0==1
      guess_vec = [guess_vec, x0];
      parameter_list{counter} = 'x0';
      counter=counter+1;
   end
   if flag_same_sigma==0
       guess_vec = [guess_vec, b];
       parameter_list{counter} = 'Ay';
       counter=counter+1;
   end
   if flag_fit_y0==1
      guess_vec = [guess_vec, y0]; 
      parameter_list{counter} = 'y0';
      counter=counter+1;
   end
   if flag_fit_correlated==1
      guess_vec = [guess_vec, c]; 
      parameter_list{counter} = 'Axy';
      counter=counter+1;
   end
   
   %Build parameters to transder into a "evaluate_2D_gaussian" function:
   %(1). Build the exponential parameters into a matrix so that it will be used in evaulate_2D_polynomial
   exponential_coeffs_mat = zeros(3,3);
   exponential_coeffs_mat(1,1) = 0; %because the constant enters through N(normalization) of the exponent:
   exponential_coeffs_mat(1,2) = (-2*a*x0-c*y0);
   exponential_coeffs_mat(2,1) = (-2*b*y0-c*x0);
   exponential_coeffs_mat(3,1) = a;
   exponential_coeffs_mat(1,3) = b;
   exponential_coeffs_mat(2,2) = c;
   new_x = linspace(1,3,20);
   [new_x,new_y] = meshgrid(new_x);
   bla = reshape(evaluate_2D_gaussian(new_x,new_y,B,N,exponential_coeffs_mat),size(new_x));
   scatter3(original_x(:),original_y(:),original_z(:));
   hold on;
   mesh(new_x,new_y,bla);
   
   if flag_lsq_or_and_nonlinear==1
      beta = guess_vec; 
   end
    
end


%Use semi-random guess if no initial lsq:
if flag_lsq_or_and_nonlinear==2
   N = median(z);
   [~,max_position] = max(z);
   x0 = x(max_position);
   y0 = y(max_position);
   a = 1/(3*(max(x)-min(x)));
   b = 1/(3*(max(y)-min(y)));
   rho = 0;
   B = median(z);
   
   %Initial search parameters for nonlinear search:
   B=min(z(:));
   N=max(z(:))-min(z(:));
   a = -(max(x)-min(x));
   b = -(max(y)-min(y));
   rho=0;
   
   %offset z:
   original_offset = min(z);
   z = z-original_offset;
   
   %Build initial parameter guess vec:
   guess_vec = [];
   if flag_fit_background==1
       guess_vec = [guess_vec, B];
   end 
   guess_vec = [guess_vec, N];
   guess_vec = [guess_vec, a];
   if flag_fit_x0==1
      guess_vec = x0; 
   end
   guess_vec = [guess_vec, b];
   if flag_fit_y0==1
      guess_vec = [guess_vec, y0]; 
   end
   if flag_fit_correlated==1
      rho = 1-c;
      guess_vec = [guess_vec, c]; 
   end
end


%NonLinear Fit:
if flag_lsq_or_and_nonlinear==2 || flag_lsq_or_and_nonlinear==3

    %Initialize parameter list:
    parameter_list = {};
    
    %Start building function call as string:
    f_str = 'f = @(params,x,y)';

    %Initialize parameter counter to keep track of how many parameters are being fitted:
    parameter_counter=1;
    parameter_list = {};

    %Fit background if wanted:
    if flag_fit_background==1
       f_str = [f_str 'params(' num2str(parameter_counter) ')+']; 
       parameter_list{parameter_counter} = 'Background';
       parameter_counter = parameter_counter+1;
    else
       %do nothing... background=0 
    end

    %Add normalization to gaussian:
    f_str = [f_str 'params(' num2str(parameter_counter) ')*exp(('];
    parameter_list{parameter_counter} = 'Normalization';
    parameter_counter = parameter_counter+1;


    %Add x^2 or (x-x0)^2 term:
    f_str = [f_str 'params(' num2str(parameter_counter) ')*'];
    parameter_list{parameter_counter} = 'Ax';
    parameter_counter = parameter_counter+1;
    if flag_fit_x0==0
       f_str = [f_str 'x.^2+'];
    else
       f_str = [f_str '(x-params(' num2str(parameter_counter) ')).^2+'];
       parameter_list{parameter_counter} = 'x0';
       parameter_x0_index = parameter_counter;
       parameter_counter = parameter_counter+1;
    end

    %Add y^2 or (y-y0)^2 term:
    if flag_same_sigma==0
        f_str = [f_str 'params(' num2str(parameter_counter) ')*'];
        parameter_list{parameter_counter} = 'Ay';
        parameter_counter = parameter_counter+1;
    else
        if flag_fit_x0 == 0 && flag_fit_background==0
            f_str = [f_str 'params(2)*'];
        elseif flag_fit_x0==1 && flag_fit_background==0
            f_str = [f_str 'params(2)*'];
        elseif flag_fit_x0==0 && flag_fit_background==1
            f_str = [f_str 'params(3)*'];
        elseif flag_fit_x0==1 && flag_fit_background==1
            f_str = [f_str 'params(3)*'];
        end
    end
    if flag_fit_y0==0
       f_str = [f_str 'y.^2'];
    else
       f_str = [f_str '(y-params(' num2str(parameter_counter) ')).^2']; 
       parameter_list{parameter_counter} = 'y0';
       parameter_y0_index = parameter_counter;
       parameter_counter = parameter_counter+1;
    end

    %Add y*x or (y-y0)(x-x0) term:
    if flag_fit_correlated==1
        f_str = [f_str '+params(' num2str(parameter_counter) ')*'];
        parameter_list{parameter_counter} = 'Axy';
        parameter_counter = parameter_counter+1;
        if flag_fit_x0==0 && flag_fit_y0==0
           f_str = [f_str 'x.*y))'];
        elseif flag_fit_x0==1 && flag_fit_y0==1
           f_str = [f_str '(y-params(' num2str(parameter_y0_index) ')).*(x-params(' num2str(parameter_x0_index) '))))']; 
           parameter_list{parameter_counter} = 'y0';
           parameter_counter = parameter_counter+1;
        elseif flag_fit_x0==0 && flag_fit_y0==1
           f_str = [f_str 'x.*(y-params(' num2str(parameter_y0_index) '))))']; 
        elseif flag_fit_x0==1 && flag_fit_y0==0
           f_str = [f_str '(x-params(' num2str(parameter_x0_index) ')).*y))']; 
        end
    else
        f_str = [f_str '))'];
    end

    eval(f_str); 
    options.Robust = 'on';
    [beta,output_vec] = fit_2D_using_matlab_1D_solvers(x,y,z,'nlinfit',f,guess_vec,options);
    if flag_lsq_or_and_nonlinear == 1 || flag_lsq_or_and_nonlinear==3
        beta(1) = beta(1) + original_offset;
    end
	1;   
end


    
