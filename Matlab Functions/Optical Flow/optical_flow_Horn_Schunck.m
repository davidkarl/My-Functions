function [u, v] = optical_flow_Horn_Schunck(mat1, ...
                                            mat2, ...
                                            alpha_smoothness_parameter, ...
                                            number_of_iterations, ...
                                            u_initial, ...
                                            v_initial, ...
                                            flag_display_flow, ...
                                            flag_display_image)

% Horn-Schunck optical flow method 
% Horn, B.K.P., and Schunck, B.G., Determining Optical Flow, AI(17), No.
% 1-3, August 1981, pp. 185-203 http://dspace.mit.edu/handle/1721.1/6337

%Get image and translate it:
% mat1 = imread('yos9.tif');
% mat2 = imread('yos10.tif');
 
% mat1 = create_speckles_of_certain_size_in_pixels(50,512,1,0);
% mat2 = shift_matrix(mat1,1,0.46,0.12);
% mat1 = abs(mat1).^2;
% mat2 = abs(mat2).^2;

%Default parameters: 
if nargin<3
    alpha_smoothness_parameter = 1;
end
if nargin<4
    number_of_iterations = 100;
end
if nargin<5 || nargin<6
    u_initial = zeros(size(mat1(:,:,1)));
    v_initial = zeros(size(mat2(:,:,1)));
elseif size(u_initial,1)==0 || size(v_initial,1)==0
    u_initial = zeros(size(mat1(:,:,1)));
    v_initial = zeros(size(mat2(:,:,1)));
end
if nargin<7
    flag_display_flow = 1;
end
if nargin<8
    flag_display_image = mat1;
end
 

%Convert images to grayscale and double:
if size(size(mat1),2)==3
    mat1 = rgb2gray(mat1);
end
if size(size(mat2),2)==3
    mat2 = rgb2gray(mat2);
end
mat1 = double(mat1);
mat2 = double(mat2);

%build gaussian filter to smooth images:
gaussian_filter_sigma = 1;
gaussian_filter_size = 2*(3*gaussian_filter_sigma);
gaussian_filter_x_vec = -(gaussian_filter_size/2) : (1+1/gaussian_filter_size) : (gaussian_filter_size/2);
gaussian_filter = (1/(sqrt(2*pi)*gaussian_filter_sigma)) * exp (-(gaussian_filter_x_vec.^2)/(2*gaussian_filter_sigma^2));

%Smooth images (understand how this damaged detection and why twice? and
%what's the difference between twice and once with sigma=2*sigma_initial):
mat1 = conv2(mat1,gaussian_filter,'same');
mat1 = conv2(mat1,gaussian_filter,'same');
mat2 = conv2(mat2,gaussian_filter,'same');
mat2 = conv2(mat2,gaussian_filter,'same');

%Estimate spatiotemporal derivatives:
flag_derivative_method = 1;
if flag_derivative_method==1
    %(1).Horn-Schunck original method
    fx = conv2(mat1,0.25* [-1 1; -1 1],'same') + conv2(mat2, 0.25*[-1 1; -1 1],'same');
    fy = conv2(mat1, 0.25*[-1 -1; 1 1], 'same') + conv2(mat2, 0.25*[-1 -1; 1 1], 'same');
    ft = conv2(mat1, 0.25*ones(2),'same') + conv2(mat2, -0.25*ones(2),'same');
elseif flag_derivative_method==2
    %(2).derivatives as in Barron
    fx = conv2(mat1,(1/12)*[-1 8 0 -8 1],'same');
    fy = conv2(mat1,(1/12)*[-1 8 0 -8 1]','same');
    ft = conv2(mat1, 0.25*ones(2),'same') + conv2(mat2, -0.25*ones(2),'same');
    fx = -fx; 
    fy = -fy;
elseif flag_derivative_method==3
    %(3).An alternative way to compute the spatiotemporal derivatives is to use simple finite difference masks.
    fx = conv2(mat1,[1 -1]);
    fy = conv2(mat1,[1; -1]);
    ft = mat2 - mat1;
end


%Set initial value for the flow vectors:
u = u_initial;
v = v_initial;

%Averaging 2D frame kernel:
kernel_2D_frame = [ 1/12 1/6 1/12 ;
                    1/6  0   1/6  ; 
                    1/12 1/6 1/12 ];

%Iterations:
for i=1:number_of_iterations
    %Compute local averages of the flow vectors:
    u_averaged = conv2(u,kernel_2D_frame,'same');
    v_averaged = conv2(v,kernel_2D_frame,'same');
    %Compute flow vectors constrained by its local average and the optical flow constraints
    u = u_averaged - ...
        (fx .* (fx.*u_averaged + fy.*v_averaged + ft) ) ./ ( alpha_smoothness_parameter^2 + fx.^2 + fy.^2); 
    v = v_averaged - ...
        (fy .* (fx.*u_averaged + fy.*v_averaged + ft) ) ./ ( alpha_smoothness_parameter^2 + fx.^2 + fy.^2);
end
u(isnan(u)) = 0;
v(isnan(v)) = 0;

%Plotting:
flag_display_flow = 0;
if flag_display_flow == 1    
    
    imagesc(mat1);
    hold on; 
    rSize=20;
    scale=3;
    
    % Enhance the quiver plot visually by showing one vector per region
    for i=1:size(u,1)
        for j=1:size(u,2)
            if floor(i/rSize)~=i/rSize || floor(j/rSize)~=j/rSize
                u(i,j)=0;
                v(i,j)=0;
            end
        end
    end
    quiver(u, v, scale, 'color', 'b', 'linewidth', 2);
    set(gca,'YDir','reverse');
        
end