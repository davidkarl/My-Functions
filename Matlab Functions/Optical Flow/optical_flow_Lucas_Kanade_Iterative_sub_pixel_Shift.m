function [ u v H ] = optical_flow_Lucas_Kanade_Iterative_sub_pixel_Shift( mat1, mat2, window_size, number_of_iterations, number_of_pyramid_levels )

%This function find the optical flow from A to B using pyramid
%representation and iteration
%To handle the pixels on the borders, a smaller window is used

%PYRE_NO : total number of pyramid levels
%ITER_NO : number of iterations at each pyramid level
%G : Gaussian kernel for smoothing


%Check Image sizes:
%If image sizes are not integer multiples of maximum downsampling ratio, then this function resizes them:
if sum( mod( size(mat1), 2^(number_of_pyramid_levels-1) ) ) ~= 0 
    mat1 = imresize( mat1, size(mat1) - mod( size(mat1), 2^(number_of_pyramid_levels-1) ) );
end
if sum( mod( size(mat2), 2^(number_of_pyramid_levels-1) ) ) ~= 0 
    mat2 = imresize( mat2, size(mat2) - mod( size(mat2), 2^(number_of_pyramid_levels-1) ) );
end    


%Form image pyramids and Initialize other variables:
half_window = (window_size-1)/2;
gaussian_filter = fspecial('gaussian',[3,3],1);
mat1_pyramids{1} = conv2(mat1,gaussian_filter,'same');
mat2_pyramids{1} = conv2(mat2,gaussian_filter,'same');
for pyramid_level_counter = 2:number_of_pyramid_levels
   mat1_pyramids{pyramid_level_counter} = impyramid(mat1_pyramids{pyramid_level_counter-1}, 'reduce');
   mat2_pyramids{pyramid_level_counter} = impyramid(mat2_pyramids{pyramid_level_counter-1}, 'reduce');
end


%Initialize the velocity field to zero at first:
if pyramid_level_counter == number_of_pyramid_levels
    u = zeros(size( mat1_pyramids{pyramid_level_counter} ));
    v = zeros(size( mat1_pyramids{pyramid_level_counter} ));
    flag_first_calculation_done = 0;
end
    

%Go over the different pyramid levels and iteratively find the proper solution:
for pyramid_level_counter = number_of_pyramid_levels:-1:1
    
    %Reflect mat1's border to allow window size:
    mat1_temp = mat1_pyramids{pyramid_level_counter};
    mat1 = zeros(size(mat1_temp)+2*half_window);
    mat1(half_window+1:size(mat1_temp,1)+half_window, half_window+1:size(mat1_temp,2)+half_window) = mat1_temp;
    for j = 1:half_window
        mat1(half_window+1 : size(mat1_temp,1)+half_window, j) = mat1_temp(1:size(mat1_temp,1), half_window+1-j);
    end
    for j = size(mat1_temp,2)+1+half_window:size(mat1_temp,2)+2*half_window
        mat1(half_window+1 : size(mat1_temp,1)+half_window, j) = mat1_temp(1:size(mat1_temp,1), 2*size(mat1_temp,2)+half_window+1-j);
    end
    for i = 1:half_window
        mat1(i, 1:size(mat1,2)) = mat1(2*half_window+1-i, 1:size(mat1,2));
    end
    for i = size(mat1_temp,1)+1+half_window:size(mat1_temp,1)+2*half_window
        mat1(i, 1:size(mat1,2)) = mat1(2*size(mat1_temp,1)+1+2*half_window-i, 1:size(mat1,2));
    end
    mat2_temp = mat2_pyramids{pyramid_level_counter};
    mat2 = mat2_pyramids{pyramid_level_counter};
    
     
    %Generating the Hessian matrices for this level
    for k = 1:number_of_iterations
        
        %Shift mat2 to former guess using interpolation:
        if flag_first_calculation_done == 0
            flag_first_calculation_done = 1;
        else            
            [x,y] = meshgrid(1:size(mat2_temp,2),1:size(mat2_temp,1));
            mat2_interpolated = interp2(mat2_temp, x+u, y+v, 'cubic');
            mat2_interpolated(isnan(mat2_interpolated)) = mat2_temp(isnan(mat2_interpolated));
        end
        
        %Reflect mat2's border to allow window size:
        mat2 = zeros(size(mat2_interpolated)+2*half_window);
        mat2(half_window+1:size(mat2_interpolated,1)+half_window, half_window+1:size(mat2_interpolated,2)+half_window) = mat2_interpolated;
        for j = 1:half_window
            mat2(half_window+1 : size(mat2_interpolated,1)+half_window, j) = mat2_interpolated(1:size(mat2_interpolated,1), half_window+1-j);
        end
        for j = size(mat2_interpolated,2)+1+half_window:size(mat2_interpolated,2)+2*half_window
            mat2(half_window+1 : size(mat2_interpolated,1)+half_window, j) = mat2_interpolated(1:size(mat2_interpolated,1), 2*size(mat2_interpolated,2)+half_window+1-j);
        end
        for i = 1:half_window
            mat2(i, 1:size(mat2,2)) = mat2(2*half_window+1-i, 1:size(mat2,2));
        end
        for i = size(mat2_interpolated,1)+1+half_window:size(mat2_interpolated,1)+2*half_window
            mat2(i, 1:size(mat2,2)) = mat2(2*size(mat2_interpolated,1)+1+2*half_window-i, 1:size(mat2,2));
        end
        
        
        %Calculate mat2 gradient for Lucas-Kanade Step:
        [Ix,Iy] = gradient(mat2);
        
        %Calculate gradient field Hessian matrix / Image moments:
        H = zeros([2,2,size(Ix)-half_window]);
        for i = 1+half_window : size(Ix,1)-half_window
            for j = 1+half_window : size(Ix,2)-half_window
                ix = Ix( i-half_window:i+half_window, j-half_window:j+half_window );
                iy = Iy( i-half_window:i+half_window, j-half_window:j+half_window );
                H(1,1,i,j) = alfa+sum(sum( ix.^2 ));
                H(2,2,i,j) = alfa+sum(sum( iy.^2 ));
                H(1,2,i,j) = sum(sum( ix .* iy ));
                H(2,1,i,j) = H(1,2,i,j);
            end
        end
        
        %Calculate temporal gradient:
        It = mat1 - mat2;
        
        %Use Lucas-Kanade to get velocity field:
        us = zeros(size(It));
        vs = zeros(size(It));
        for i = 1+half_window : size(It,1)-half_window
            for j = 1+half_window : size(It,2)-half_window
                %Get sub-blocks:
                ix = Ix( i-half_window:i+half_window, j-half_window:j+half_window );
                iy = Iy( i-half_window:i+half_window, j-half_window:j+half_window );
                it = It( i-half_window:i+half_window, j-half_window:j+half_window );
                %Set up and Solve Lucas-Kanade:
                b(1,1) = sum(sum( it .* ix ));
                b(2,1) = sum(sum( it .* iy ));
                x = H(:,:,i,j) \ b;
                %Assign solution to flow field:
                us(i,j) = x(1);
                vs(i,j) = x(2);
            end
        end
        

        %Add found solutions to iteratively updated final solutions:
        us = us(half_window+1:size(us,1)-half_window, half_window+1:size(us,2)-half_window);
        vs = vs(half_window+1:size(vs,1)-half_window, half_window+1:size(vs,2)-half_window);   
        u = u + us;
        v = v + vs;
    end %End of iterations loop
       
    %Scale flow field (SEEMS WEIRD....SHOULDN'T THE SCALING BE FOR us ????)
    if pyramid_level_counter ~= 1 
        u = 2 * imresize(u,size(u)*2,'bilinear');
        v = 2 * imresize(v,size(v)*2,'bilinear');
    end

    
end %End of pyramid levels loop



end
