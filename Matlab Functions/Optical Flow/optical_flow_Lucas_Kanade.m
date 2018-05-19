function [vx,vy] = optical_flow_Lucas_Kanade(mat1,mat2,block_size)
% clear all;

% mat1 = create_speckles_of_certain_size_in_pixels(50,512,1,0);
% mat2 = shift_matrix(mat1,1,0.46,0);
% mat1 = abs(mat1).^2;
% mat2 = abs(mat2).^2;

% x_blk_size = floor(.04*size(mat1,2));
% y_blk_size = floor(.04*size(mat1,1));
x_blk_size = block_size;
y_blk_size = block_size;

%Calculate change in image with respect to position and time:
[Ix,Iy] = gradient(mat1);
It = mat2 - mat1;
[ny,nx] = size(Ix);
sub_domain_counter = 1;
x1 = 1;
x2 = x1 + x_blk_size;
x_number_of_blocks = floor((nx-x_blk_size)/x_blk_size);
y_number_of_blocks = floor((ny-y_blk_size)/y_blk_size);
vx = zeros(size(mat1));
vy = zeros(size(mat1));
x = zeros(size(mat1));
y = zeros(size(mat1));


%loop through image:
for ix = 1:x_number_of_blocks
    y1 = 1;
    y2 = y1 + y_blk_size;
    for iy = 1:y_number_of_blocks
        
        % select a sub-domain from gradient and difference image to perform calculation on:
        Ix_block = Ix(y1:y2,x1:x2);
        Iy_block = Iy(y1:y2,x1:x2);
        It_block = It(y1:y2,x1:x2);
        
        % Cast problem as linear equation and solve in a lsqr sense
        % This approach is known as the Lucas-Kanade Method
        % A*u = f
        % A'*A*u = A'*f
        % u = inv(A'*A)*A*f
        A = [Ix_block(:) , Iy_block(:)];
        b = -It_block(:);
        A = A(1:1:end,:);
        b = b(1:1:end);
        P = pinv(A'*A);
        u = P*A'*b;
        
        % realtive velocities from current sub-domain:
        vx(x1,y1) = u(1);
        vy(x1,y1) = u(2);
         
        % calculate mid point of sub-domain:
        y(x1,y1) = (x1+x2)/2;
        x(x1,y1) = (y1+y2)/2;
        
        sub_domain_counter = sub_domain_counter+1;
        
        % get the y range of the new block:
        y1 = y1 + y_blk_size + 1;
        y2 = y1 + y_blk_size;
        
        % make sure you don't exceed the image size in the y direction:
        if y2 > ny
            y2  = ny;
        end
    end
    
    % get the x range of the new block
    x1 = x1 + x_blk_size + 1;
    x2 = x1 + x_blk_size;

    % make sure you don't exceed the image size in the x direction
    if x2 > nx
        x2 = nx;
    end
end



%Plotting:
flag_display_flow = 0;
if flag_display_flow == 1    
    
    imagesc(mat1);
    hold on;
    rSize=20;
    scale=3;
    
    % Enhance the quiver plot visually by showing one vector per region
    for i=1:size(vx,1)
        for j=1:size(vx,2)
            if floor(i/rSize)~=i/rSize || floor(j/rSize)~=j/rSize
                vx(i,j)=0;
                vy(i,j)=0;
            end
        end
    end
    quiver(vx, vy, scale, 'color', 'b', 'linewidth', 2);
    set(gca,'YDir','reverse');
        
end
