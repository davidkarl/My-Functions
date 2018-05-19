function [u,v] = optical_flow_Lucas_Kanade_Iterative_Integer_Shift(mat1,mat2)

%Data acquisition
mat1=single((imread('yos_img_10.pgm')));
mat2=single((imread('yos_img_11.pgm')));

%parameters : levels number, window size, iterations number, regularization
number_of_pyramid_levels = 3;
window_size = 9;
iterations_counter = 1;
alpha_regularization = 0.001;
hw = floor(window_size/2);

%pyramids creation:
pyramid1 = mat1;
pyramid2 = mat2;

%initialize pyramid levels:
for pyramid_level_counter = 2:number_of_pyramid_levels
    %Use matlab built in function to reduce image size, look into function
    %as this is not trivial (for instance it doesn't do anti aliasing etc'):
    mat1 = impyramid(mat1, 'reduce');
    mat2 = impyramid(mat2, 'reduce');
    
    %Get new (reduced) image sizes:
    [M1,N1] = size(mat1);
    [M2,N2] = size(mat2);
    
    %Get current pyramid level images:
    pyramid1(M1, N1, pyramid_level_counter) = mat1;
    pyramid2(M2, N2, pyramid_level_counter) = mat2;
end


%Processing all levels:
for pyramid_level_counter = 1:number_of_pyramid_levels
   
    %current pyramid:
    M1 = size(pyramid1,1) / 2^(number_of_pyramid_levels - pyramid_level_counter);
    N1 = size(pyramid1,2) / 2^(number_of_pyramid_levels - pyramid_level_counter);
    M2 = size(pyramid2,1) / 2^(number_of_pyramid_levels - pyramid_level_counter);
    N2 = size(pyramid2,2) / 2^(number_of_pyramid_levels - pyramid_level_counter);
    mat1 = pyramid1(1:M1, 1:N1, (number_of_pyramid_levels - pyramid_level_counter)+1);
    mat2 = pyramid2(1:M2, 1:N2, (number_of_pyramid_levels - pyramid_level_counter)+1);
       
    %Initialize velocity estimations:
    if pyramid_level_counter == 1
        u = zeros(size(mat1));
        v = zeros(size(mat1));
    else  
        %resizing:
        %AGAIN - is bilinear really the smartest way?
        u = 2 * imresize(u,size(u)*2,'bilinear');
        v = 2 * imresize(v,size(v)*2,'bilinear');
    end
    
    
    %Estimation refinment loop:
    for r = 1:iterations_counter
        
        %Round flow field to shift blocks being matched by integer pixels: 
        u = round(u);
        v = round(v);
        
        %every pixel loop
        for i = 1+hw:size(mat1,1)-hw
            for j = 1+hw:size(mat2,2)-hw
                
                sub_block_mat1 = mat1(i-hw:i+hw, j-hw:j+hw);
                
                %moved patch:
                low_row = i-hw+v(i,j);
                high_row = i+hw+v(i,j);
                low_column = j-hw+u(i,j);
                high_column = j+hw+u(i,j);
                
                if (low_row < 1)||(high_row > size(mat1,1))||(low_column < 1)||(high_column > size(mat1,2))
                    %Regularized least square processing
                else
                    %Get mat2 current sub block:
                    sub_block_mat2 = mat2(low_row:high_row, low_column:high_column);
                    
                    %Smooth gradients:
                    fx = conv2(sub_block_mat1, 0.25* [-1 1; -1 1]) + conv2(sub_block_mat2, 0.25*[-1 1; -1 1]);
                    fy = conv2(sub_block_mat1, 0.25* [-1 -1; 1 1]) + conv2(sub_block_mat2, 0.25*[-1 -1; 1 1]);
                    ft = conv2(sub_block_mat1, 0.25*ones(2)) + conv2(sub_block_mat2, -0.25*ones(2));
                    
                    %Get relevant parts:
                    Fx = fx(2:window_size-1,2:window_size-1)';
                    Fy = fy(2:window_size-1,2:window_size-1)';
                    Ft = ft(2:window_size-1,2:window_size-1)';
                    
                    %Get image moments:
                    A = [Fx(:) , Fy(:)];
                    G = A'*A;
                    G(1,1) = G(1,1)+alpha_regularization; 
                    G(2,2) = G(2,2)+alpha_regularization;
                    
                    %Get new solution;
                    U = 1/(G(1,1)*G(2,2)-G(1,2)*G(2,1))*[G(2,2),-G(1,2);-G(2,1),G(1,1)]*A'*-Ft(:);
                    
                    %Refine:
                    u(i,j) = u(i,j) + U(1); 
                    v(i,j) = v(i,j) + U(2);
                end
            end
        end
    end
end

%resizing:
u = u(window_size:size(u,1)-window_size+1,window_size:size(u,2)-window_size+1);
v = v(window_size:size(v,1)-window_size+1,window_size:size(v,2)-window_size+1);

%colormap display
figure(1)
RGB1=showmap3(u,v,5);
imshow(RGB1);