function [U, V] = optical_flow_Lucas_Kanade_smoothed(mat1,mat2, ...
                        m200_previous,m020_previous,m110_previous,m101_previous,m011_previous,...
                        dx_previous,dy_previous,dt_previous)

%Resolution of video and flow field [Height Width]:
flow_resolution = [24,24];
temporal_smoothing_factor_moments = 0.2;
temporal_smoothing_factor_gradient = sqrt(temporal_smoothing_factor_moments);

%Build gaussian filter:
gaussian_filter_sigma = 1.8;
gaussian_filter_size = round(5*gaussian_filter_sigma);
gaussian_filter_size = gaussian_filter_size + mod(gaussian_filter_size-1,2);
gaussian_filter_axis = -(gaussian_filter_size-1)/2 : 1 :(gaussian_filter_size-1)/2;
gaussian_filter = exp( -(gaussian_filter_axis).^2/(2*gaussian_filter_sigma^2) );
gaussian_filter = gaussian_filter / sum(gaussian_filter(:));

%Initialize variables:
if isempty(gg) || nargin <3
    m200 = zeros(flow_resolution,'single');
    m020 = zeros(flow_resolution,'single');
    m110 = zeros(flow_resolution,'single');
    m101 = zeros(flow_resolution,'single');
    m011 = zeros(flow_resolution,'single');
    dx = zeros(size(mat2),'single');
    dy = zeros(size(mat2),'single');
    dt = zeros(size(mat2),'single');
end


%Calculate 3D gradient:
g_kernel = [0.2163,   0.5674,   0.2163];
%understand whether this is an easier approach to the bidirectional estimation...i don't think so because dt should be reversed
mats_sum = mat2 + mat1; 
dx = temporal_smoothing_factor_gradient*dx_previous + ...
    (1-temporal_smoothing_factor_gradient)*conv2(mats_sum(:,[2:end end ]) - mats_sum(:,[1 1:(end-1)]),g_kernel','same');
dy = temporal_smoothing_factor_gradient*dy_previous + ...
    (1-temporal_smoothing_factor_gradient)*conv2(mats_sum([2:end end],: ) - mats_sum([1 1:(end-1)],:),g_kernel ,'same');
dt = temporal_smoothing_factor_gradient*dt_previous + ...
    (1-temporal_smoothing_factor_gradient)*2*conv2(g_kernel,g_kernel,mat2 - mat1,'same'); %why the factor 2 - because mats_sum is sum of two matrices

%Tikhinov Constant (Etot = E1 + 1/2*TC*|V|^2):
TC = single(5^2); 
% TC = single((110-10*g.gamma)^g.gamma); 

m = 550;
gam = 0.2;
N_regularization_and_stabilization = (dx.^2+dy.^2 +dt.^2  + m*(6*gam)^2+eps)/m;                
% nor = 1;%((dx.^2 + dy.^2).^(1-g.gamma))+1;

%MOMENT CALCULATIONS:
%(1).make gamma-corrected elementwise product
m200 = (dx.^2)./N_regularization_and_stabilization;
m020 = (dy.^2)./N_regularization_and_stabilization;
m110 = (dx.*dy)./N_regularization_and_stabilization;
m101 = (dx.*dt)./N_regularization_and_stabilization;
m011 = (dy.*dt)./N_regularization_and_stabilization;
%(2).smooth with large seperable gaussian filter (spatial integration)
m200 = conv2(gg,gg,m200,'same');
m020 = conv2(gg,gg,m020,'same');
m110 = conv2(gg,gg,m110,'same');
m101 = conv2(gg,gg,m101,'same');
m011 = conv2(gg,gg,m011,'same');
%(3).downsample to specified resolution (imresizeNN function is found in "helperFunctions"):
m200 =  imresizeNN(m200 ,flow_resolution);
m020 =  imresizeNN(m020 ,flow_resolution);
m110 =  imresizeNN(m110 ,flow_resolution);
m101 =  imresizeNN(m101 ,flow_resolution);
m011 =  imresizeNN(m011 ,flow_resolution);
%(4).add Tikhonov constant if a diagonal element (for m200, m020):
m200 =  m200 + TC;
m020 =  m020 + TC;
%(5) update the moment output (recursive filtering, temporal integration)
m200 = temporal_smoothing_factor_moments*m200_previous + (1-temporal_smoothing_factor_moments)*m200;
m020 = temporal_smoothing_factor_moments*m020_previous + (1-temporal_smoothing_factor_moments)*m020;
m110 = temporal_smoothing_factor_moments*m110_previous + (1-temporal_smoothing_factor_moments)*m110;
m101 = temporal_smoothing_factor_moments*m101_previous + (1-temporal_smoothing_factor_moments)*m101;
m011 = temporal_smoothing_factor_moments*m011_previous + (1-temporal_smoothing_factor_moments)*m011;
%L2: Implement the vectorized formulation of the solver:
U =(-m101.*m020 + m011.*m110)./(m020.*m200 - m110.^2);%-2*TC^2/3);
V =( m101.*m110 - m011.*m200)./(m020.*m200 - m110.^2);%-2*TC^2/3);

 


%Horn Schunck!!!!!!!!:
%Parameters:
alpha_smoothness_parameters = single(10); %closely related to the TC, however here its Etot=E1+1/2*eta*(u_averaged-u)^2
max_number_of_iterations = 100;

%make the kernel, normalize it to EnR. A lower EnR pushes down the
%magnitude of no texture regions faster. As the maximum nof iterations
%go up, EnR should be set closer to 1.
EnR = min(1,  0.92 + 0.1*max_number_of_iterations/100);
kern=[1       sqrt(2) 1       ; ...
    sqrt(2) 0       sqrt(2) ; ...
    1       sqrt(2) 1      ];
kern = single(EnR*kern/sum(kern(:)));

%explicit algorithm from Horn and Schunks original paper:
%(1).I CAN USE ANY INITIAL GUESS I WANT, IF I DO NOTHING THEN IT'S THE ABOVE
%DERIVED LK SOLUTION, I CAN DO OTHER THINGS.
%(2).I NEED TO FORMULATE AND CODE AN ITERATIVE SOLUTION WHICH MOVES THE
%IMAGE AROUND EACH ITERATION OR DOES AN INTERPOLATION OF THE LAGRANGIAN
%MINIMUM OR MATS DIFFERENCE ERROR MIN TO PERHAPSE HAVE HIGHER RESOLUTION
for i=1:max_number_of_iterations
    uAvg=conv2(U,kern,'same');      
    vAvg=conv2(V,kern,'same');
    U = uAvg - dx.*(dx.*uAvg + dy.*vAvg  + dt)./(alpha_smoothness_parameters.^2 + dx.^2 +dy.^2);
    V = vAvg - dy.*(dx.*uAvg + dy.*vAvg  + dt)./(alpha_smoothness_parameters.^2 + dx.^2 +dy.^2);
end

end %END OF MAIN FUNCTION!!!!



function [outputImage] = imresizeNN(inputImage, newSize)
%%%%%%% imresizeNN(inputImage, newSize) is identical to built in 
%%%%%%% imresize(inputImage, newSize, 'nearest'), but is much faster
oldSize = size(inputImage);  
scale = newSize./oldSize;    

% Compute a resampled set of indices:
outputImage = inputImage(...
    min(round(((1:newSize(1))-0.5)./scale(1)+0.5),oldSize(1)),...
    min(round(((1:newSize(2))-0.5)./scale(2)+0.5),oldSize(2))     );
end


function imOut = imresizeBL(imIn, newSize)
    kernel = @gauss;
    kernel_width = 20;

    oldSize = size(imIn);  
    scale   = newSize./oldSize(1:2);    

    % Determine which dimension to resize first.
    [~, order] = sort(scale);

    % Calculate interpolation weights and indices for each dimension.
    	%%%% beware that in-place computations dont work for mex calls!
    
    k = order(1);
    [weights, indices] = contributions(oldSize(k), ...
        newSize(k), scale(k), kernel, kernel_width, 1);
%     note! if you dont have imresizemex in your matlab path, then find
%     it in the image processing toolbox. Its in a private folder,
%     accessible from "imresize.m"
    imOut = imresizemex(imIn, weights', indices', k);
% save; error
    k = order(2);
    [weights, indices] = contributions(oldSize(k), ...
        newSize(k), scale(k), kernel, kernel_width, 1);
    imOut = imresizemex(imOut, weights', indices', k);
	
end

function [weights, indices] = contributions(in_length, out_length, ...
                                            scale, kernel, ...
                                            kernel_width, antialiasing)

if (scale < 1) && (antialiasing)
    % Use a modified kernel to simultaneously interpolate and
    % antialias.
    h = @(x) scale * kernel(scale * x);
    kernel_width = kernel_width / scale;
else
    % No antialiasing; use unmodified kernel.
    h = kernel;
end

% Output-space coordinates.
x = (1:out_length)';

% Input-space coordinates. Calculate the inverse mapping such that 0.5
% in output space maps to 0.5 in input space, and 0.5+scale in output
% space maps to 1.5 in input space.
u = x/scale + 0.5 * (1 - 1/scale);

% What is the left-most pixel that can be involved in the computation?
left = floor(u - kernel_width/2);

% What is the maximum number of pixels that can be involved in the
% computation?  Note: it's OK to use an extra pixel here; if the
% corresponding weights are all zero, it will be eliminated at the end
% of this function.
P = ceil(kernel_width) + 2;

% The indices of the input pixels involved in computing the k-th output
% pixel are in row k of the indices matrix.
indices = bsxfun(@plus, left, 0:P-1);

% The weights used to compute the k-th output pixel are in row k of the
% weights matrix.
weights = h(bsxfun(@minus, u, indices));

% Normalize the weights matrix so that each row sums to 1.
weights = bsxfun(@rdivide, weights, sum(weights, 2));

% Clamp out-of-range indices; has the effect of replicating end-points.
indices = min(max(1, indices), in_length);

% If a column in weights is all zero, get rid of it.
kill = find(~any(weights, 1));
if ~isempty(kill)
    weights(:,kill) = [];
    indices(:,kill) = [];
end
end

function f = triangle(x)
f = (x+1) .* ((-1 <= x) & (x < 0)) + (1-x) .* ((0 <= x) & (x <= 1));
end

function f = gauss(x)
f = exp(-(2.7/50)*x.^2);
end
