function [vx_mat vy_mat] = optical_flow_Chan_Vo(mat1, mat2, ...
                                                block_size, ...
                                                flag_do_median_filtering, ...
                                                flag_search_algorithm, ...
                                                flag_check_bidirectional)

% mat1 = create_speckles_of_certain_size_in_pixels(50,512,1,0);
% mat2 = shift_matrix(mat1,1,0.25,0);
% mat1 = abs(mat1).^2;
% mat2 = abs(mat2).^2;
 
%Default parameters:
% block_size = 10;
% flag_search_algorithm = 1;
% flag_check_bidirectional = 1;
search_range = 1;
 
%Crop the images so that image size is a multiple of BlockSize:
M = floor(size(mat2, 1)/block_size)*block_size;
N = floor(size(mat2, 2)/block_size)*block_size;
mat2 = mat2(1:M,1:N,:);
mat1 = mat1(1:M,1:N,:);

%Enlarge the image boundaries by BlockSize/2 pixels
mat2 = padarray(mat2, [block_size/2 block_size/2], 'replicate');
mat1 = padarray(mat1, [block_size/2 block_size/2], 'replicate');

%Pad zeros to images to fit SearchLimit
mat2 = padarray(mat2, [search_range, search_range]);
mat1 = padarray(mat1, [search_range, search_range]);

%Set parameters:
[M,N,C] = size(mat2);
half_block_size = floor(block_size/2);
sub_block_axis = -half_block_size:half_block_size-1;
xc_range = search_range+half_block_size+1 : block_size : N-(search_range+half_block_size);
yc_range = search_range+half_block_size+1 : block_size : M-(search_range+half_block_size);

%Initialize motion estimates mat:
vx_mat = zeros(length(yc_range), length(xc_range));
vy_mat = zeros(length(yc_range), length(xc_range));

 
%Main Loop:
for sub_block_y_counter = 1:length(yc_range)
    for sub_block_x_counter = 1:length(xc_range)
        
        %Get mat1 sub block:
        x_center = xc_range(sub_block_x_counter);
        y_center = yc_range(sub_block_y_counter);
        current_sub_block_x_range = x_center + sub_block_axis;
        current_sub_block_y_range = y_center + sub_block_axis;
        mat1_sub_block = mat1(current_sub_block_y_range , current_sub_block_x_range , :);

        %Initialize sum of absolute difference min:
        sum_of_absolute_difference_min = 1e6; %something large
        
        %Rejection:
        if (y_center<=search_range+half_block_size)||(y_center>=M-(search_range+half_block_size))
            error('Can you set yc >%3g pixels from the boundary? \n',search_range+half_block_size);
        end
        if (x_center<=search_range+half_block_size)||(x_center>=N-(search_range+half_block_size))
            error('Can you set xc >%3g pixels from the boundary? \n',search_range+half_block_size);
        end
        
        %Choose either one of the followings:
%         flag_search_algorithm = 2;
         
        if flag_search_algorithm==1   
            %Full Search Loop:
            for sub_block_y_shift = -search_range:1:search_range 
                for sub_block_x_shift = -search_range:1:search_range
                    
                    %Get current reference image block:
                    xt = x_center + sub_block_x_shift; 
                    yt = y_center + sub_block_y_shift;
                    current_sub_block_x_range2 = xt + sub_block_axis;
                    current_sub_block_y_range2 = yt + sub_block_axis;
                    mat2_sub_block = mat2(current_sub_block_y_range2, current_sub_block_x_range2, :);
                    
                    %Get sum of absolute difference between reference image block and current image block:
                    sum_of_absolute_difference = sum(abs(mat1_sub_block(:) - mat2_sub_block(:))) / (block_size^2);
                    
                    %Check if a new minimum in sum_of_absolute_difference is reached:
                    if sum_of_absolute_difference < sum_of_absolute_difference_min
                        sum_of_absolute_difference_min  = sum_of_absolute_difference;
                        x_min = xt;
                        y_min = yt;
                    end
                    
                    %Motion Vector (integer part):
                    MVx_int = x_center - x_min;
                    MVy_int = y_center - y_min;
                end
            end
        elseif flag_search_algorithm==2
            %Log Search Loop;
            x0 = x_center;
            y0 = y_center;
            
            %Main Loop:
            LevelMax = 2;
            LevelLimit = zeros(1,LevelMax+1);
            for k = 1:LevelMax
                LevelLimit(k+1) = max(2^(floor(log2(search_range))-k+1),1);
                c = 2.^(0:log2(LevelLimit(k+1)));
                c(c+sum(LevelLimit(1:k))>search_range) = [];
                x_range = zeros(1,2*length(c)+1);
                x_range(1) = 0;
                x_range(2:2:2*length(c)) = c;
                x_range(3:2:2*length(c)+1) = -c;
                y_range = x_range;
                
                for i = 1:length(y_range)
                    for j = 1:length(x_range)
                        if sum_of_absolute_difference_min>1e-3
                            xt = x0 + x_range(j);
                            yt = y0 + y_range(i);
                            mat2_sub_block = mat2(yt+sub_block_axis, xt+sub_block_axis, :);
                            sum_of_absolute_difference = sum(abs(mat1_sub_block(:) - mat2_sub_block(:)))/(block_size^2);
                            if sum_of_absolute_difference < sum_of_absolute_difference_min
                                sum_of_absolute_difference_min  = sum_of_absolute_difference;
                                x_min = xt;
                                y_min = yt;
                            end
                        else
                            sum_of_absolute_difference_min = 0;
                            x_min = xc;
                            y_min = yc;
                        end
                        
                    end
                end
                
                x0 = x_min;
                y0 = y_min;
            end %Level loop end
            
            %Motion Vector (integer part):
            MVx_int = x_center - x_min;
            MVy_int = y_center - y_min;
            
        end %Search Algorithm End 
        
        
        %Get optimum integer shift block:
        mat2_sub_block = mat2(y_min+sub_block_axis, x_min+sub_block_axis, :);
        
        %Taylor series approximation sub-pixel solution:
        %define linear equation parameters:
        [dfx , dfy] = gradient(mat1_sub_block);
        a = sum(dfx(:).^2);
        b = sum(dfx(:).*dfy(:));
        d = sum(dfy(:).^2);
        z = mat2_sub_block - mat1_sub_block;
        p = sum(z(:).*dfx(:));
        q = sum(z(:).*dfy(:));
        %build linear equation matrices:
        A = [a b; b d];
        rhs = [p;q];
        %solve linear equation:
        if cond(A)>1e6
            Taylor_sol = [0 0]';
        else
            Taylor_sol = A\rhs;
        end
        
        if flag_check_bidirectional == 1
            [dfx , dfy] = gradient(mat2_sub_block);
            a = sum(dfx(:).^2);
            b = sum(dfx(:).*dfy(:));
            d = sum(dfy(:).^2);
            z = mat1_sub_block - mat2_sub_block;
            p = sum(z(:).*dfx(:));
            q = sum(z(:).*dfy(:));
            %build linear equation matrices:
            A = [a b; b d];
            rhs = [p;q];
            %solve linear equation:
            if cond(A)>1e6
                Taylor_sol2 = [0 0]';
            else
                Taylor_sol2 = A\rhs;
            end
            
            Taylor_sol = (Taylor_sol + (-Taylor_sol2))/2;
        end 
         
        %Motion Vector (fractional part):
        MVx_frac = Taylor_sol(1); 
        MVy_frac = Taylor_sol(2);
                
        % Motion Vector (overall)
        MVx = MVx_int + MVx_frac; 
        MVy = MVy_int + MVy_frac;

        %Update motion estimates mats:
        vx_mat(sub_block_y_counter,sub_block_x_counter) = MVx;
        vy_mat(sub_block_y_counter,sub_block_x_counter) = MVy;
    end
end
  
%Maximize MV:
vx_mat(vx_mat> search_range) =  search_range;
vx_mat(vx_mat<-search_range) = -search_range;
vy_mat(vy_mat> search_range) =  search_range;
vy_mat(vy_mat<-search_range) = -search_range;


%Median filter to clean up MV:
if flag_do_median_filtering == 1
    vx_mat = medfilt2(vx_mat, [3 3]);
    vy_mat = medfilt2(vy_mat, [3 3]);
end
% quiver(vx_mat,vy_mat); 

%Plotting:
flag_display_flow = 0;
if flag_display_flow == 1    
    
    imagesc(mat1);
    hold on;
    rSize=20;
    scale=3;
    
    % Enhance the quiver plot visually by showing one vector per region
    for i=1:size(vx_mat,1)
        for j=1:size(vx_mat,2)
            if floor(i/rSize)~=i/rSize || floor(j/rSize)~=j/rSize
                vx_mat(i,j)=0;
                vy_mat(i,j)=0;
            end
        end
    end
    quiver(vx_mat, vy_mat, scale, 'color', 'b', 'linewidth', 2);
    set(gca,'YDir','reverse');
        
end

end






