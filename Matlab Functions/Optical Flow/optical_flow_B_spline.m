function [Ireg,Grid_B_spline_control_points,Grid_spacing,M_Global_transformation_matrix,B,F] = ...
                                                                            optical_flow_B_spline(mat1,mat2,Options)
% This function image_registration is the most easy way to register two
% 2D or 3D images both affine and nonrigidly.
%
% Features:
% - It can be used with images from different type of scans or modalities.
% - It uses both a rigid transform and a nonrigid b-spline grid transform.
% - It uses grid refinement
% - It can be used with images of different sizes.
% - The function will automaticaly detect if the images can be registered
% with the sum of squared pixel distance (SSD), or when mutual information
% must be used as image similarity measure.
%
% Note: Compile the c-files with compile_c_files to allow 3D, and for more
%   more speed in 2D.
% 
% [Ireg,Grid,Spacing,M,B,F] = image_registration(Imoving,Istatic,Options);
%
% Inputs,
%   Imoving : The image which will be registerd
%   Istatic : The image on which Imoving will be registered
%   Options : Registration options, see help below
%
% Outputs,
%   Ireg : The registered moving image
%   Grid: The b-spline controlpoints, can be used to transform another
%       image in the same way: I=bspline_transform(Grid,I,Spacing);
%   Spacing: The uniform b-spline knot spacing
%	M : The affine transformation matrix
%   B : The backwards transformation fields  of the pixels in
%       x,y and z direction seen from the  static image to the moving image.
%		in 2D, Bx=B(:,:,1); By=B(:,:,2);
%   F : The (approximated) forward transformation fields of the pixels in
%       x, y and z direction seen from the moving image to the static image.
%		in 2D, Fx=F(:,:,1); Fy=F(:,:,2);
%       (See the function backwards2forwards)
%
% Options,
%   Options.Similarity: Similarity measure (error) used can be set to:
%               sd : Squared pixel distance
%               mi : Normalized (Local) mutual information
%               d, gd, gc, cc, pi, ld : see image_difference.m.
%   Options.Registration:
%               Rigid    : Translation, Rotation
%               Affine   : Translation, Rotation, Shear, Resize
%               NonRigid : B-spline grid based local registration
%               Both     : Nonrigid and Affine (Default)
%   Options.Penalty: Thin sheet of metal smoothness penalty, default in 2D 1e-3 ,
%				default in 3D, 1e-5
%               if set to zero the registration will take a shorter time, but
%               will give a more distorted transformation field.
%   Options.Interpolation: Linear (default) or Cubic, the final result is
%               always cubic interpolated.
%   Options.MaxRef : Maximum number of grid refinements steps (default 2)
%   Options.Grid: Initial B-spline controlpoints grid, if not defined is initalized
%               with an uniform grid. (Used for in example while
%               registering
%               a number of movie frames)
%   Options.Spacing: Spacing of initial B-spline grid in pixels 1x2 [sx sy] or 1x3 [sx sy sz]
%               sx, sy, sz must be powers of 2, to allow grid refinement.
%   Options.MaskMoving: Image which is transformed in the same way as Imoving and
%               is multiplied with the individual pixel errors
%               before calculation of the te total (mean) similarity
%               error. In case of measuring the mutual information similariy
%               the zero mask pixels will be simply discarded.
%   Options.MaskStatic: Also a Mask but is used  for Istatic
%   Options.Verbose: Display Debug information 0,1 or 2
%
% Corresponding points / landmarks Options,
%   Options.Points1: List N x 2 or N x 3 of landmarks x,y(,z) in Imoving image
%   Options.Points2: List N x 2 or N x 3 of landmarks x,y(,z) in Istatic image, in which
%                     every row correspond to the same row with landmarks
%                     in Points1.
%   Options.PStrength: List Nx1 with the error strength used between the
%                     corresponding points, (lower the strenght of the landmarks
%                     if less sure of point correspondence).
%



%i will only deal with 2D:
IS3D = false; 

%assign default options:
default_options = struct('Similarity',[],...
                        'Registration','Both',...
                        'Penalty',1e-3,...
                        'MaxRef',2,...
                        'Grid',[],...
                        'Spacing',[],...
                        'MaskMoving',[],...
                        'MaskStatic',[],...
                        'Verbose',2,...
                        'Points1',[],...
                        'Points2',[],...
                        'PStrength',[],...
                        'Interpolation','Linear',...
                        'Scaling',[1 1]);


%Decide whether to use default options or given options:
if(~exist('Options','var')) 
    Options = default_options;
else
    tags = fieldnames(default_options);
    for i=1:length(tags),
        if(~isfield(Options,tags{i})), Options.(tags{i})=default_options.(tags{i}); end
    end
    if(length(tags)~=length(fieldnames(Options))),
        warning('image_registration:unknownoption','unknown options found');
    end
end

%Set parameters:
%   Options.Similarity: Similarity measure (error) used can be set to:
%   sd : Squared pixel distance
%   mi : Normalized (Local) mutual information
%   d, gd, gc, cc, pi, ld : see image_difference.m.
similarity_measure_type = Options.Similarity;

%   Options.Grid: Initial B-spline controlpoints grid, if not defined is initalized
%   with an uniform grid. (Used for in example while registering a number of movie frames)
Grid_B_spline_control_points = Options.Grid; 

%   Options.Spacing: Spacing of initial B-spline grid in pixels 1x2 [sx sy] or 1x3 [sx sy sz]
%   sx, sy, sz must be powers of 2, to allow grid refinement(!)
Grid_spacing = Options.Spacing;

%   Options.MaskMoving: Image which is transformed in the same way as Imoving and
%   is multiplied with the individual pixel errors before calculation of the te total (mean) similarity
%   error. In case of measuring the mutual information similariy the zero mask pixels will be simply discarded.
mat1_error_mask = Options.MaskMoving; 

%   Options.MaskStatic: Also a Mask but is used  for Istatic:
mat2_error_mask = Options.MaskStatic;

%   Options.Points1: List N x 2 or N x 3 of landmarks x,y(,z) in Imoving image:
mat1_landmark_points = Options.Points1; 

%   Options.Points2: List N x 2 or N x 3 of landmarks x,y(,z) in Istatic image, in which
%    every row correspond to the same row with landmarks in Points1.
mat2_landmark_points = Options.Points2; 

%   Options.PStrength: List Nx1 with the error strength used between the
%   corresponding points, (lower the strength of the landmarks if less sure of point correspondence).
landmark_point_strengths = Options.PStrength;




% Start time measurement:
%   Options.Verbose: Display Debug information 0,1 or 2
if(Options.Verbose>0) 
    tic; 
end

%Scale input images into the range of [0,1]:
Iclass = class(mat1);
Imin = min( min(mat2(:)) , min(mat1(:)) );
Imax = max( max(mat2(:)) , max(mat1(:)) );
mat1 = (mat1-Imin)/(Imax-Imin);
mat2 = (mat2-Imin)/(Imax-Imin);

%Resize the moving image to fit the static image:
if(sum(size(mat2)-size(mat1))~=0)
    %Resize the moving image to fit the static image:
    mat1 = imresize(mat1,[size(mat2,1),size(mat2,2)],'bicubic');
    %Resize the MASKmoving image to fit MASKstatic:
    if ~isempty(mat1_error_mask)
        mat1_error_mask = imresize(mat1_error_mask,[size(mat2,1),size(mat2,2)],'bicubic');
    end
end

%Detect if the mutual information or pixel distance can be used as similarity measure. 
%By comparing the histograms.
if isempty(similarity_measure_type)
    mat1_normalized_histogram = hist(mat1(:),(1/60)*[0.5:60])./numel(mat1);
    mat2_normalized_histogram = hist(mat2(:),(1/60)*[0.5:60])./numel(mat2);
    if(sum(abs(mat1_normalized_histogram(:)-mat2_normalized_histogram(:)))>0.25),
        similarity_measure_type = 'mi';
        if(Options.Verbose>0), disp('Multi Modalities, Mutual information is used'); drawnow; end
    else
        similarity_measure_type = 'sd';
        if(Options.Verbose>0), disp('Same Modalities, Pixel Distance is used'); drawnow; end
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%GET GLOBAL TRANSFORMATION MATRIX M:

%Register the moving image affine to the static image:
%   Options.Registration:
%               Rigid    : Translation, Rotation
%               Affine   : Translation, Rotation, Shear, Resize
%               NonRigid : B-spline grid based local registration
%               Both     : Nonrigid and Affine (Default)
if ~strcmpi(Options.Registration(1),'N') %if it's NOT nonrigid
    %RIGID, AFFINE, AND BOTH
    
    %Make smooth for fast affine registration:
    gaussian_filter_sigma = 2.5;
    gaussian_filter_xy_size = [10,10];
    mat1_smoothed = imfilter(mat1,fspecial('gaussian',gaussian_filter_xy_size,gaussian_filter_sigma));
    mat2_smoothed = imfilter(mat2,fspecial('gaussian',gaussian_filter_xy_size,gaussian_filter_sigma));
    
    %Get parameters scaling and initial affine parameters:
    if(strcmpi(Options.Registration(1),'R'))
        %(1).RIGID - Translation, Rotation
        %Parameter scaling of the Translation and Rotation:
        scale = [1,1,1];
        %Set initial affine parameters:
        x = [0,0,0];
    elseif(strcmpi(Options.Registration(1),'A'))
        %(2). AFFINE - Translation, Rotation, Shear, Resize:
        scale = [1,1,0.01,0.01,0.01,0.01,0.01];
        %Set initial affine parameters:
        x = [0,0,0,1,1,0,0];
    elseif(strcmpi(Options.Registration(1),'B'))
        %(3). Both - Nonrigid and Affine (Default):
        scale = [1,1,0.01,0.01,0.01,0.01,0.01];
        %Set initial affine parameters:
        x = [0,0,0,1,1,0,0];
    end
    
    %Choose interpolation mode (Linear and Cubic):
    if Options.Interpolation(1)=='L'
        %Linear:
        interpolation_mode = 0;
    else
        %Cubic:
        interpolation_mode = 2;
    end
    
    %Register Affine with 3 scale spaces:
    for refinement_iteration_counter=1:3
        
        
        if(refinement_iteration_counter==1)
            %First Iteration:
            mat1_smoothed_resized = imresize(mat1_smoothed,0.25);
            mat2_smoothed_resized = imresize(mat2_smoothed,0.25);
            mat1_landmark_points_resized = mat1_landmark_points*0.25;
            mat2_landmark_points_resized = mat2_landmark_points*0.25;
            landmark_point_strenghts_resized = landmark_point_strengths;
            
            if ~isempty(mat1_error_mask)
                mat1_error_mask_resized = imresize(mat1_error_mask,0.25);
            else
                mat1_error_mask_resized = [];
            end
            if ~isempty(mat2_error_mask)
                mat2_error_mask_resized = imresize(mat2_error_mask,0.25);
            else
                mat2_error_mask_resized = [];
            end
            
        elseif(refinement_iteration_counter==2)
            %Second Iteration:
            x(1:2) = x(1:2)*2;
            mat1_smoothed_resized = imresize(mat1_smoothed,0.5);
            mat2_smoothed_resized = imresize(mat2_smoothed,0.5);
            mat1_landmark_points_resized = mat1_landmark_points*0.5;
            mat2_landmark_points_resized = mat2_landmark_points*0.5;
            landmark_point_strenghts_resized = landmark_point_strengths;
            
            
            if ~isempty(mat1_error_mask)
                mat1_error_mask_resized = imresize(mat1_error_mask,0.5);
            else mat1_error_mask_resized = [];
            end
            if ~isempty(mat2_error_mask)
                mat2_error_mask_resized = imresize(mat2_error_mask,0.5);
            else mat2_error_mask_resized = [];
            end
            
            
        elseif(refinement_iteration_counter==3)
            %Third Iteration:
            x(1:2) = x(1:2)*2;
            mat1_smoothed_resized = mat1;
            mat2_smoothed_resized = mat2;
            mat1_error_mask_resized = mat1_error_mask;
            mat2_error_mask_resized = mat2_error_mask;
            mat1_landmark_points_resized = mat1_landmark_points;
            mat2_landmark_points_resized = mat2_landmark_points;
            landmark_point_strenghts_resized = landmark_point_strengths;
        end
        
        
        %Minimizer parameters:
        %Use struct because expanded optimset is part of the Optimization Toolbox:
        optim = struct('GradObj','on',...
                       'GoalsExactAchieve',1,...
                       'Display','off',...
                       'StoreN',10,...
                       'HessUpdate','lbfgs',...
                       'MaxIter',100,...
                       'MaxFunEvals',1000,...
                       'TolFun',1e-7,...
                       'DiffMinChange',1e-3); %maybe change to smaller then 1e-3 ????

        if Options.Verbose>0
            optim.Display = 'iter';
        end
        
        %Scale the translation, resize and rotation parameters to scaled optimizer values
        x = x./scale;
        %Find the Afffine deformation:
        x = fminlbfgs(@(x)affine_registration_error(x,...
                                                    scale,...
                                                    mat1_smoothed_resized,...
                                                    mat2_smoothed_resized,...
                                                    similarity_measure_type,...
                                                    Grid_B_spline_control_points,...
                                                    Grid_spacing,...
                                                    mat1_error_mask_resized,...
                                                    mat2_error_mask_resized,...
                                                    mat1_landmark_points_resized,...
                                                    mat2_landmark_points_resized,...
                                                    landmark_point_strenghts_resized,...
                                                    interpolation_mode)                         ,x,optim);
        %ReScale the translation, resize and rotation parameters to the real values
        x = x.*scale;
        
    end %END OF ITERATIONS LOOP
    
    
    %Use the found solution for minimum affine transform registration error to get the GLOBAL T matrix:
    %Initialize transformation vectors:
    translation_vector = x(1:2);
    rotation_clockwise_vector = x(3);
    if length(x)>=4
        resize_vector = x(4:5);
        shear_vector = x(6:7);
    else
        resize_vector = [1,1]; 
        shear_vector = [0,0]; 
    end

    %Calculate affine transformation matrix:
    M_translation = ...
       [1 0 translation_vector(1);
	    0 1 translation_vector(2);
	    0 0 1];
    M_resize = ...
       [resize_vector(1) 0    0;
	    0    resize_vector(2) 0;
	    0    0    1];
    M_rotation = ...
        [ cos(rotation_clockwise_vector) sin(rotation_clockwise_vector) 0;
	     -sin(rotation_clockwise_vector) cos(rotation_clockwise_vector) 0;
	      0 0 1];
    M_shear = ...
        [1    shear_vector(1) 0;
	     shear_vector(2) 1    0;
	     0    0    1];
     M_Global_transformation_matrix = M_translation * M_resize * M_rotation * M_shear;
      
else
    %if it IS NON-RIGID:
    M_Global_transformation_matrix = [];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Make the initial b-spline registration grid:

%(1). Determine max number of grid refinement iterations and get Grid_spacing:
if isempty(Grid_B_spline_control_points)
    
    if isempty(Options.Spacing)
        %Calculate max refinements steps:
        max_number_of_grid_refinement_iterations = min(floor(log2([size(mat1,1),size(mat1,2)]/4)));
        
        %set b-spline grid spacing in x and y direction:
        Grid_spacing = [2^max_number_of_grid_refinement_iterations,2^max_number_of_grid_refinement_iterations];
    else
        %set b-spline grid spacing in x and y direction:
        Grid_spacing = round(Options.Spacing);
        t = Grid_spacing; 
        max_number_of_grid_refinement_iterations = 0; 
        while( (nnz(mod(t,2))==0) && (nnz(t<8)==0) )  %as long as both grid spacings are even and larger then 8
            t = t/2; 
            max_number_of_grid_refinement_iterations = max_number_of_grid_refinement_iterations + 1; 
        end
    end
else
    max_number_of_grid_refinement_iterations = 0;
    t = Grid_spacing;
    while mod(t,2) == 0
        t = t/2;
        max_number_of_grid_refinement_iterations = max_number_of_grid_refinement_iterations+1;
    end
end
%Limit refinements steps to user input:
if(Options.MaxRef < max_number_of_grid_refinement_iterations) 
    max_number_of_grid_refinement_iterations = Options.MaxRef; 
end


%(2).Make the Initial b-spline registration grid if not given beforehand:
input_image_sizes = [size(mat1,1),size(mat1,2)];
if isempty(Grid_B_spline_control_points)
    
    dx = Grid_spacing(1); 
    dy = Grid_spacing(2);
    [X,Y] = ndgrid(-dx:dx:(input_image_sizes(1)+(dx*2)),-dy:dy:(input_image_sizes(2)+(dy*2)));
    Grid_B_spline_control_points = ones(size(X,1),size(X,2),2);
    Grid_B_spline_control_points(:,:,1) = X;
    Grid_B_spline_control_points(:,:,2) = Y;
end

%(3).Correct for M_Global_transformation_matrix:
if ~strcmpi(Options.Registration(1),'N')
    %other then non-rigid:
    
    %Calculate center of the image:
    mean = input_image_sizes/2;
    
    %Make center of the image coordinates 0,0:
    xd = Grid_B_spline_control_points(:,:,1) - mean(1);
    yd = Grid_B_spline_control_points(:,:,2) - mean(2);
    
    %Calculate the rigid transformed coordinates:
    Grid_B_spline_control_points(:,:,1) = mean(1) + ...
        M_Global_transformation_matrix(1,1) * xd + ...
        M_Global_transformation_matrix(1,2) * yd + ...
        M_Global_transformation_matrix(1,3) * 1;
    Grid_B_spline_control_points(:,:,2) = mean(2) + ...
        M_Global_transformation_matrix(2,1) * xd + ...
        M_Global_transformation_matrix(2,2) * yd + ...
        M_Global_transformation_matrix(2,3) * 1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    
    


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Register the moving image nonrigid to the static image:

if ( strcmpi(Options.Registration(1),'N') || strcmpi(Options.Registration(1),'B') )
    %NON-RIGID OR BOTH (RIGID AND NON-RIGID):
    
    %Non-rigid b-spline grid registration
    if(Options.Verbose>0)
        disp('Start non-rigid b-spline grid registration');
        drawnow;
    end
    if (Options.Verbose>0)
        disp(['Current Grid size : ', num2str(size(Grid_B_spline_control_points,1)), 'x', num2str(size(Grid_B_spline_control_points,2))]);
        drawnow;
    end
    
    %Set registration options:
    options.type = similarity_measure_type;
    options.penaltypercentage = Options.Penalty;
    options.interpolation = Options.Interpolation;
    options.scaling = Options.Scaling;
    options.verbose = false;
    
    %Enable forward instead of central gradient incase of error measure is pixel distance
    if strcmpi(similarity_measure_type,'sd')
        options.centralgrad = false;
    end
    
    %Reshape O_trans from a matrix to a vector:
    Grid_B_spline_size = size(Grid_B_spline_control_points);
    Grid_B_spline_control_points = Grid_B_spline_control_points(:);
    
    %Make smooth images for fast registration without local minimums:
    gaussian_smoothing_filter_size = round( 0.25*( size(mat1,1)/size(Grid_B_spline_control_points,1) + ...
                          size(mat2,2)/size(Grid_B_spline_control_points,2) ) );
    gaussian_smoothing_filter_sigma = gaussian_smoothing_filter_size/5;
    mat1_smoothed = imfilter(mat1,fspecial('gaussian',[gaussian_smoothing_filter_size,gaussian_smoothing_filter_size],gaussian_smoothing_filter_sigma));
    mat2_smoothed = imfilter(mat2,fspecial('gaussian',[gaussian_smoothing_filter_size,gaussian_smoothing_filter_size],gaussian_smoothing_filter_sigma));
    resize_factor = 2^(max_number_of_grid_refinement_iterations-1);
    
    %Resize the mats and mask to the image size used in the registration:
    if ~isempty(mat1_error_mask)
        mat1_error_mask_resized = imresize(mat1_error_mask,1/resize_factor);
    else
        mat1_error_mask_resized = [];
    end
    if ~isempty(mat2_error_mask)
        mat2_error_mask_resized = imresize(mat2_error_mask,1/resize_factor);
    else
        mat2_error_mask_resized = [];
    end
    mat1_smoothed_resized = imresize(mat1_smoothed,1/resize_factor);
    mat2_smoothed_resized = imresize(mat2_smoothed,1/resize_factor);
    
    %Use struct because expanded optimset is part of the Optimization Toolbox.
    optim = struct('GradObj','on',...
                   'GoalsExactAchieve',0,...
                   'StoreN',10,...
                   'HessUpdate','lbfgs',...
                   'Display','off',...
                   'MaxIter',100,...
                   'DiffMinChange',0.001,...
                   'DiffMaxChange',1,...
                   'MaxFunEvals',1000,...
                   'TolX',0.005,...
                   'TolFun',1e-8);
    
    if Options.Verbose>0
        optim.Display = 'iter';
    end
    
    %Start the b-spline nonrigid registration optimizer:
    Grid_spacing_resized = Grid_spacing/resize_factor;
    mat1_landmark_points_resized = mat1_landmark_points/resize_factor;
    mat2_landmark_points_resized = mat2_landmark_points/resize_factor;
    landmark_point_strengths_resized = landmark_point_strengths;
    Grid_B_spline_control_points = resize_factor * ...
                    fminlbfgs(@(x)bspline_registration_gradient(x,...
                                                                Grid_B_spline_size,...
                                                                Grid_spacing_resized,...
                                                                mat1_smoothed_resized,...
                                                                mat2_smoothed_resized,...
                                                                options,...
                                                                mat1_error_mask_resized,...
                                                                mat2_error_mask_resized,...
                                                                mat1_landmark_points_resized,...
                                                                mat2_landmark_points_resized,...
                                                                landmark_point_strengths_resized),...
                                                                Grid_B_spline_control_points/resize_factor , optim);
    
    %Reshape O_trans from a vector to a matrix:
    Grid_B_spline_control_points = reshape(Grid_B_spline_control_points,Grid_B_spline_size);
    
    %Refinement loop:
    for refinement_iteration_counter = 1:max_number_of_grid_refinement_iterations
        if (Options.Verbose>0), disp('Registration Refinement'); drawnow; end
        
        %Refine image transformation grid of 1D b-splines with use of spliting matrix:
        %Initial parameters:
        mat1_size = size(mat1);
        Grid_spacing = Grid_spacing/2;
        
        %Refine B-spline grid in the x-direction:
        O_newA = zeros( (2*(size(Grid_B_spline_control_points,1)-2)-1) + 2,...
                         size(Grid_B_spline_control_points,2), 2 );
        i = 1:size(Grid_B_spline_control_points,1)-3;
        j = 1:size(Grid_B_spline_control_points,2);
        h = 1:2;
        [I,J,H] = ndgrid(i,j,h);
        I = I(:); 
        J = J(:); 
        H = H(:);
        
        ind = sub2ind(size(Grid_B_spline_control_points),I,J,H);
        P0 = Grid_B_spline_control_points(ind);
        P1 = Grid_B_spline_control_points(ind+1);
        P2 = Grid_B_spline_control_points(ind+2);
        P3 = Grid_B_spline_control_points(ind+3);
        Pnew(:,1) = (4/8)*(P0+P1);
        Pnew(:,2) = (1/8)*(P0+6*P1+P2);
        Pnew(:,3) = (4/8)*(P1+P2);
        Pnew(:,4) = (1/8)*(P1+6*P2+P3);
        Pnew(:,5) = (4/8)*(P2+P3);
        
        ind = sub2ind(size(O_newA),1+2*(I-1),J,H);
        O_newA(ind) = Pnew(:,1);
        O_newA(ind+1) = Pnew(:,2);
        O_newA(ind+2) = Pnew(:,3);
        O_newA(ind+3) = Pnew(:,4);
        O_newA(ind+4) = Pnew(:,5);
        
        %Refine B-spline grid in the y-direction:
        O_newB = zeros(size(O_newA,1),...
                     ((size(O_newA,2)-2)*2-1)+2,2);
        i = 1:size(O_newA,2)-3; 
        j = 1:size(O_newA,1); 
        h = 1:2;
        [J,I,H] = ndgrid(j,i,h);
        I = I(:); 
        J = J(:); 
        H = H(:);
        
        ind = sub2ind(size(O_newA),J,I,H);
        P0 = O_newA(ind);
        P1 = O_newA(ind+size(O_newA,1));
        P2 = O_newA(ind+size(O_newA,1)*2);
        P3 = O_newA(ind+size(O_newA,1)*3);
        Pnew(:,1) = (4/8)*(P0+P1);
        Pnew(:,2) = (1/8)*(P0+6*P1+P2);
        Pnew(:,3) = (4/8)*(P1+P2);
        Pnew(:,4) = (1/8)*(P1+6*P2+P3);
        Pnew(:,5) = (4/8)*(P2+P3);
        
        ind = sub2ind(size(O_newB),J,1+((I-1)*2),H);
        O_newB(ind) = Pnew(:,1);
        O_newB(ind+size(O_newB,1)) = Pnew(:,2);
        O_newB(ind+2*size(O_newB,1)) = Pnew(:,3);
        O_newB(ind+3*size(O_newB,1)) = Pnew(:,4);
        O_newB(ind+4*size(O_newB,1)) = Pnew(:,5);
        
        %Set the final refined matrix:
        Grid_B_spline_control_points = O_newB;
        
        %Make sure a new uniform grid will have the same dimensions (crop)
        dx = Grid_spacing(1); 
        dy = Grid_spacing(2);
        X = ndgrid(-dx:dx:(mat1_size(1)+(dx*2)),-dy:dy:(mat1_size(2)+(dy*2)));
        Grid_B_spline_control_points = Grid_B_spline_control_points(1:size(X,1),1:size(X,2),1:2);
        %END GRID REFINEMENT PROCESS   
            

        %Make smooth images for fast registration without local minimums
        gaussian_smoothing_filter_size = round(0.25*(size(mat1_smoothed,1)/size(Grid_B_spline_control_points,1)+size(mat2_smoothed,2)/size(Grid_B_spline_control_points,2)));
        mat1_smoothed = imfilter(mat1,fspecial('gaussian',[gaussian_smoothing_filter_size gaussian_smoothing_filter_size],gaussian_smoothing_filter_size/5));
        mat2_smoothed = imfilter(mat2,fspecial('gaussian',[gaussian_smoothing_filter_size gaussian_smoothing_filter_size],gaussian_smoothing_filter_size/5));
        resize_factor = 2^(max_number_of_grid_refinement_iterations - refinement_iteration_counter - 1);
        
        %No smoothing in last registration step:
        if refinement_iteration_counter == max_number_of_grid_refinement_iterations
            mat1_smoothed = mat1;
            mat2_smoothed = mat2;
            resize_factor = 1;
            optim.TolX = 0.03;
        end
        
        %Show current grid size:
        if Options.Verbose>0
            disp(['Current Grid size : ', num2str(size(Grid_B_spline_control_points,1)), 'x', num2str(size(Grid_B_spline_control_points,2))]);
            drawnow;
        end
        
        %Reshape O_trans from a matrix to a vector:
        Grid_B_spline_size = size(Grid_B_spline_control_points);
        Grid_B_spline_control_points = Grid_B_spline_control_points(:);
        
        %Resize the mask to the image size used in the registration
        if ~isempty(mat1_error_mask)
            mat1_error_mask_resized = imresize(mat1_error_mask,1/resize_factor);
        else
            mat1_error_mask_resized = [];
        end
        if ~isempty(mat2_error_mask)
            mat2_error_mask_resized = imresize(mat2_error_mask,1/resize_factor);
        else
            mat2_error_mask_resized = [];
        end
        
        
        %Start the b-spline nonrigid registration optimizer:
        mat1_smoothed_resized = imresize(mat1_smoothed,1/resize_factor);
        mat2_smoothed_resized = imresize(mat2_smoothed,1/resize_factor);
        
        Grid_spacing_resized = Grid_spacing/resize_factor;
        mat1_landmark_points_resized = mat1_landmark_points/resize_factor;
        mat2_landmark_points_resized = mat2_landmark_points/resize_factor;
        landmark_point_strengths_resized = landmark_point_strengths;
        Grid_B_spline_control_points = resize_factor * ...
                                         fminlbfgs(@(x)bspline_registration_gradient(x,...
                                                                                     Grid_B_spline_size,...
                                                                                     Grid_spacing_resized,...
                                                                                     mat1_smoothed_resized,...
                                                                                     mat2_smoothed_resized,...
                                                                                     options,...
                                                                                     mat1_error_mask_resized,...
                                                                                     mat2_error_mask_resized,...
                                                                                     mat1_landmark_points_resized,...
                                                                                     mat2_landmark_points_resized,...
                                                                                     landmark_point_strengths_resized),...
                                                                                     Grid_B_spline_control_points/resize_factor, optim);

        %Reshape O_trans from a vector to a matrix:
        Grid_B_spline_control_points = reshape(Grid_B_spline_control_points,Grid_B_spline_size);
    end %END OF REFINEMENT ITERATIONS LOOP
    
end %END OF NON RIGID / BOTH RIGID+NON RIGID CONDITION



%
%Transform the input image with the found optimal grid.
%Check if spacing has integer values
if sum(Grid_spacing-floor(Grid_spacing))>0
    error('Spacing must be a integer');
end

%Define Grid parameters:
interpolation_mode = 3;
Grid_point_coordinates_x = Grid_B_spline_control_points(:,:,1);
Grid_point_coordinates_y = Grid_B_spline_control_points(:,:,2);
Grid_point_spacing_x = Grid_spacing(1);
Grid_point_spacing_y = Grid_spacing(2);

%Make all x,y indices:
[x,y] = ndgrid(0:size(mat1,1)-1,0:size(mat1,2)-1);

%Calulate the transformation of all image coordinates by the b-sline grid:
O_trans(:,:,1) = Grid_point_coordinates_x; 
O_trans(:,:,2) = Grid_point_coordinates_y;
Spacing = [Grid_point_spacing_x,Grid_point_spacing_y];
X = [x(:),y(:)];

%Make row vectors of input coordinates:
x2 = X(:,1); 
y2 = X(:,2);

%Make polynomial look up tables:
Bu = zeros(4,Spacing(1));
Bv = zeros(4,Spacing(2));
Bdu = zeros(4,Spacing(1));
Bdv = zeros(4,Spacing(2));

x = 0 : Spacing(1)-1;
u = (x/Spacing(1)) - floor(x/Spacing(1));
Bu(0*Spacing(1)+x+1) = (1-u).^3/6;
Bu(1*Spacing(1)+x+1) = ( 3*u.^3 - 6*u.^2 + 4)/6;
Bu(2*Spacing(1)+x+1) = (-3*u.^3 + 3*u.^2 + 3*u + 1)/6;
Bu(3*Spacing(1)+x+1) = u.^3/6;

y = 0 : Spacing(2)-1;
v = (y/Spacing(2)) - floor(y/Spacing(2));
Bv(0*Spacing(2)+y+1) = (1-v).^3/6;
Bv(1*Spacing(2)+y+1) = ( 3*v.^3 - 6*v.^2 + 4)/6;
Bv(2*Spacing(2)+y+1) = (-3*v.^3 + 3*v.^2 + 3*v + 1)/6;
Bv(3*Spacing(2)+y+1) = v.^3/6;

Bdu(0*Spacing(1)+x+1) = -(u - 1).^2/2;
Bdu(1*Spacing(1)+x+1) = (3*u.^2)/2 - 2*u;
Bdu(2*Spacing(1)+x+1) = -(3*u.^2)/2 + u + 1/2;
Bdu(3*Spacing(1)+x+1) = u.^2/2;

Bdv(0*Spacing(2)+y+1) = -(v - 1).^2/2;
Bdv(1*Spacing(2)+y+1) = (3*v.^2)/2 - 2*v;
Bdv(2*Spacing(2)+y+1) = -(3*v.^2)/2 + v + 1/2;
Bdv(3*Spacing(2)+y+1) = v.^2/2;

Bdu = Bdu./Spacing(1);
Bdv = Bdv./Spacing(2);

%Calculate the indexes need to loop up the B-spline values:
u_index = mod(x2,Spacing(1));
v_index = mod(y2,Spacing(2));

i = floor(x2/Spacing(1)); % (first row outside image against boundary artefacts)
j = floor(y2/Spacing(2));

%This part calculates the coordinates of the pixel which will be transformed to the current x,y pixel:
Ox = O_trans(:,:,1);
Oy = O_trans(:,:,2);
Tlocalx = 0; 
Tlocaly = 0;
Tlocaldxx = 0;
Tlocaldyy = 0;
Tlocaldxy = 0;
Tlocaldyx = 0;

a = zeros(size(X,1),4);
b = zeros(size(X,1),4);
ad = zeros(size(X,1),4);
bd = zeros(size(X,1),4);

IndexO1l = zeros(size(X,1),4);
IndexO2l = zeros(size(X,1),4);
Check_bound1 = false(size(X,1),4);
Check_bound2 = false(size(X,1),4);

for r = 0:3
    a(:,r+1) = Bu(r*Spacing(1)+u_index(:)+1);
    b(:,r+1) = Bv(r*Spacing(2)+v_index(:)+1);
    ad(:,r+1) = Bdu(r*Spacing(1)+u_index(:)+1);
    bd(:,r+1) = Bdv(r*Spacing(2)+v_index(:)+1);

    IndexO1l(:,r+1) = (i+r);
    IndexO2l(:,r+1) = (j+r);
    Check_bound1(:,r+1) = (IndexO1l(:,r+1)<0)|(IndexO1l(:,r+1)>(size(O_trans,1)-1));
    Check_bound2(:,r+1) = (IndexO2l(:,r+1)<0)|(IndexO2l(:,r+1)>(size(O_trans,2)-1));
end

for l=0:3,
    for m=0:3,
        IndexO1 = IndexO1l(:,l+1);
        IndexO2 = IndexO2l(:,m+1);
        Check_bound = Check_bound1(:,l+1)|Check_bound1(:,m+1);
        IndexO1(Check_bound) = 1;
        IndexO2(Check_bound) = 1;
        Check_bound_inv = double(~Check_bound);
        
        ab = a(:,l+1) .* b(:,m+1);
        abx = ad(:,l+1) .* b(:,m+1);
        aby = a(:,l+1) .* bd(:,m+1);
        
        c = Ox(IndexO1(:)+IndexO2(:)*size(Ox,1)+1);
        Tlocalx = Tlocalx + Check_bound_inv(:).*ab.*c;
        Tlocaldxx = Tlocaldxx + Check_bound_inv(:).*abx.*c;
        Tlocaldxy = Tlocaldxy + Check_bound_inv(:).*aby.*c;
        
        c = Oy(IndexO1(:)+IndexO2(:)*size(Oy,1)+1);
        Tlocaly = Tlocaly + Check_bound_inv(:).*ab.*c;
        Tlocaldyx = Tlocaldyx + Check_bound_inv(:).*abx.*c;
        Tlocaldyy = Tlocaldyy + Check_bound_inv(:).*aby.*c;
    end
end
Tlocal(:,1) = Tlocalx(:);
Tlocal(:,2) = Tlocaly(:);
Dlocal = (Tlocaldxx).*(Tlocaldyy) - Tlocaldyx.*Tlocaldxy;
Tx = reshape(Tlocal(:,1),[size(Iin,1) size(Iin,2)])-x;
Ty = reshape(Tlocal(:,2),[size(Iin,1) size(Iin,2)])-y;
B(:,:,1) = Tx;
B(:,:,2) = Ty;


Interpolation = 'bicubic';
Boundary = 'zero';
Ireg = image_interpolation(Iin,Tlocal(:,1),Tlocal(:,2),Interpolation,Boundary);










        
   









%Make the forward transformation fields from the backwards
if nargout>5 
    F = backwards2forwards(B); 
end

%Convert the double registered image to the class and range of the input images:
Ireg = Ireg*(Imax-Imin)+Imin;


%End time measurement:
if(Options.Verbose>0) 
    toc
end












function Iout = image_interpolation(Iin,Tlocalx,Tlocaly,Interpolation,Boundary,ImageSize)
% This function is used to transform an 2D image, in a backwards way with an
% transformation image.
%
%   Iout = image_interpolation(Iin,Tlocalx,Tlocaly,Interpolation,Boundary,ImageSize)
%
% inputs,
%	   Iin : 2D greyscale or color input image
%	   Tlocalx,Tlocaly : (Backwards) Transformation images for all image pixels
%	   Interpolation:
%       'nearest'    - nearest-neighbor interpolation
%       'bilinear'   - bilinear interpolation
%       'bicubic'    - cubic interpolation; the default method
%      Boundary:
%       'zero'       - outside input image are implicilty assumed to be zero
%       'replicate'  - Input array values outside the bounds of the array
%                      are assumed to equal the nearest array border value
%	(optional)
%	   ImageSize:    - Size of output image
% outputs,
%  	   Iout : The transformed image
%
% Function is written by D.Kroon University of Twente (September 2010)

ImageSize = [size(Iin,1),size(Iin,2)];
if(ndims(Iin)==2)
    lo=1; %grayscale
else
    lo=3; %RGB
end


xBas0=floor(Tlocalx);
yBas0=floor(Tlocaly);
tx=Tlocalx-xBas0;
ty=Tlocaly-yBas0;

% Determine the t vectors
vec_tx0= 0.5; vec_tx1= 0.5*tx; vec_tx2= 0.5*tx.^2; vec_tx3= 0.5*tx.^3;
vec_ty0= 0.5; vec_ty1= 0.5*ty; vec_ty2= 0.5*ty.^2;vec_ty3= 0.5*ty.^3;

% t vector multiplied with 4x4 bicubic kernel gives the to q vectors
vec_qx0= -1.0*vec_tx1 + 2.0*vec_tx2 - 1.0*vec_tx3;
vec_qx1=  2.0*vec_tx0 - 5.0*vec_tx2 + 3.0*vec_tx3;
vec_qx2=  1.0*vec_tx1 + 4.0*vec_tx2 - 3.0*vec_tx3;
vec_qx3= -1.0*vec_tx2 + 1.0*vec_tx3;

vec_qy0= -1.0*vec_ty1 + 2.0*vec_ty2 - 1.0*vec_ty3;
vec_qy1=  2.0*vec_ty0 - 5.0*vec_ty2 + 3.0*vec_ty3;
vec_qy2=  1.0*vec_ty1 + 4.0*vec_ty2 - 3.0*vec_ty3;
vec_qy3= -1.0*vec_ty2 + 1.0*vec_ty3;

% Determine 1D neighbour coordinates
xn0=xBas0-1; xn1=xBas0; xn2=xBas0+1; xn3=xBas0+2;
yn0=yBas0-1; yn1=yBas0; yn2=yBas0+1; yn3=yBas0+2;


% limit indexes to boundaries
check_xn0=(xn0<0)|(xn0>(size(Iin,1)-1));
check_xn1=(xn1<0)|(xn1>(size(Iin,1)-1));
check_xn2=(xn2<0)|(xn2>(size(Iin,1)-1));
check_xn3=(xn3<0)|(xn3>(size(Iin,1)-1));
check_yn0=(yn0<0)|(yn0>(size(Iin,2)-1));
check_yn1=(yn1<0)|(yn1>(size(Iin,2)-1));
check_yn2=(yn2<0)|(yn2>(size(Iin,2)-1));
check_yn3=(yn3<0)|(yn3>(size(Iin,2)-1));
xn0=min(max(xn0,0),size(Iin,1)-1);
xn1=min(max(xn1,0),size(Iin,1)-1);
xn2=min(max(xn2,0),size(Iin,1)-1);
xn3=min(max(xn3,0),size(Iin,1)-1);
yn0=min(max(yn0,0),size(Iin,2)-1);
yn1=min(max(yn1,0),size(Iin,2)-1);
yn2=min(max(yn2,0),size(Iin,2)-1);
yn3=min(max(yn3,0),size(Iin,2)-1);


Iout=zeros([ImageSize(1:2) lo]);
for i=1:lo; % Loop incase of RGB
    Iin_one=Iin(:,:,i);

            % Get the intensities
            Iy0x0=Iin_one(1+xn0+yn0*size(Iin,1));Iy0x1=Iin_one(1+xn1+yn0*size(Iin,1));
            Iy0x2=Iin_one(1+xn2+yn0*size(Iin,1));Iy0x3=Iin_one(1+xn3+yn0*size(Iin,1));
            Iy1x0=Iin_one(1+xn0+yn1*size(Iin,1));Iy1x1=Iin_one(1+xn1+yn1*size(Iin,1));
            Iy1x2=Iin_one(1+xn2+yn1*size(Iin,1));Iy1x3=Iin_one(1+xn3+yn1*size(Iin,1));
            Iy2x0=Iin_one(1+xn0+yn2*size(Iin,1));Iy2x1=Iin_one(1+xn1+yn2*size(Iin,1));
            Iy2x2=Iin_one(1+xn2+yn2*size(Iin,1));Iy2x3=Iin_one(1+xn3+yn2*size(Iin,1));
            Iy3x0=Iin_one(1+xn0+yn3*size(Iin,1));Iy3x1=Iin_one(1+xn1+yn3*size(Iin,1));
            Iy3x2=Iin_one(1+xn2+yn3*size(Iin,1));Iy3x3=Iin_one(1+xn3+yn3*size(Iin,1));
            
            % Set pixels outside the image
            Iy0x0(check_yn0|check_xn0)=0;Iy0x1(check_yn0|check_xn1)=0;
            Iy0x2(check_yn0|check_xn2)=0;Iy0x3(check_yn0|check_xn3)=0;
            Iy1x0(check_yn1|check_xn0)=0;Iy1x1(check_yn1|check_xn1)=0;
            Iy1x2(check_yn1|check_xn2)=0;Iy1x3(check_yn1|check_xn3)=0;
            Iy2x0(check_yn2|check_xn0)=0;Iy2x1(check_yn2|check_xn1)=0;
            Iy2x2(check_yn2|check_xn2)=0;Iy2x3(check_yn2|check_xn3)=0;
            Iy3x0(check_yn3|check_xn0)=0;Iy3x1(check_yn3|check_xn1)=0;
            Iy3x2(check_yn3|check_xn2)=0;Iy3x3(check_yn3|check_xn3)=0;
        
            
            % Combine the weighted neighbour pixel intensities
            Iout_one=vec_qy0.*(vec_qx0.*Iy0x0+vec_qx1.*Iy0x1+vec_qx2.*Iy0x2+vec_qx3.*Iy0x3)+...
                vec_qy1.*(vec_qx0.*Iy1x0+vec_qx1.*Iy1x1+vec_qx2.*Iy1x2+vec_qx3.*Iy1x3)+...
                vec_qy2.*(vec_qx0.*Iy2x0+vec_qx1.*Iy2x1+vec_qx2.*Iy2x2+vec_qx3.*Iy2x3)+...
                vec_qy3.*(vec_qx0.*Iy3x0+vec_qx1.*Iy3x1+vec_qx2.*Iy3x2+vec_qx3.*Iy3x3);
    
    Iout(:,:,i)=reshape(Iout_one, ImageSize);
end









