function [output] = return_shifts_between_speckle_patterns(mat1,mat2,spacing,cross_correlation_algorithm,shift_algorithm,upsampling_images,upsampling_cross_correlation,flag_check_only_ROI_3)
% clear all;
% clc;
flag_get_max=0;
% N=100;
% spacing=0.5;
% speckle_size_in_pixels = 4;
% [speckle_pattern]=create_speckles_of_certain_size_in_pixels(speckle_size_in_pixels,N,1);
% shiftx = 0.2;
% shifty = 0;
% upsampling_images=1;
% upsampling_cross_correlation=1;
% cross_correlation_algorithm = 3;
% shift_algorithm = 4;
% speckle_pattern_shifted = shift_matrix(speckle_pattern,spacing,shiftx,shifty);
% % speckle_pattern = pixelize_field(speckle_pattern,100,1,2,N);
% % speckle_pattern_shifted = pixelize_field(speckle_pattern_shifted,100,1,2,N);
% % N=N/2;
% speckle_pattern = abs(speckle_pattern).^2;
% speckle_pattern_shifted = abs(speckle_pattern_shifted).^2;
% mat1=speckle_pattern;
% mat2=speckle_pattern_shifted;
% flag_check_only_ROI_3=1;
N=size(mat1,1); %i assume N is even!!!!

%SHIFT_ALGORITHM IS A VECTOR!!!! IN ORDER TO RETURN SEVERAL SURFACE FITS
%INSTEAD OF REPEATING THE CROSS CORRELATION ALGORITHM

%upsample images:
N_image = N*upsampling_images;
if upsampling_images>1 
   mat1 = interpft(mat1,N_image,1);
   mat1 = interpft(mat1,N_image,2);
   mat2 = interpft(mat2,N_image,1);
   mat2 = interpft(mat2,N_image,2);
end

%calculate cross correlation by the proper algorithm of calculation:
% cross_correlation_algorithm = 3;
if cross_correlation_algorithm==1 %fft
    cross_correlation = corr2_ft(mat1/norm(mat1(:)),mat2/norm(mat2(:)),spacing); 
elseif cross_correlation_algorithm==2 %normalized regular cross correlation
    cross_correlation = conv2(mat1-mean(mat1(:)),fliplr(flipud(mat2-mean(mat2(:)))),'same');
elseif cross_correlation_algorithm==3 %dan's algorithm
    flag_calculation = 1;
    calculation_count=3;
    while flag_calculation==1
        [cross_correlation] = calculate_properly_subtracted_cross_correlation_cells(mat1,mat2,calculation_count);
        [max_row,max_col] = return_max_position(cross_correlation);
        middle_index = (calculation_count+1)/2;
        if flag_check_only_ROI_3==1
           flag_calculation=0;
           max_row = 2;
           max_col = 2;
        elseif (max_row>1 && max_row<calculation_count && max_col>1 && max_col<calculation_count)
           %maximum is not on boundary and surface fitting can will be successful
           flag_calculation = 0; 
        else
           calculation_count = calculation_count + 2; 
        end
    end
end  
  
%upsample cross correlation if necessary and proper:
N_cross_correlation = N*upsampling_images*upsampling_cross_correlation;
cross_correlation_spacing = spacing/upsampling_images/upsampling_cross_correlation;
if upsampling_cross_correlation>1 && ~(cross_correlation_algorithm==3)
    cross_correlation = interpft(cross_correlation,N_cross_correlation,1);
    cross_correlation = interpft(cross_correlation,N_cros_correlation,2);
end
 
%normalize cross_correlation: 
% cross_correlation = abs(cross_correlation)/max(cross_correlation(:));
   
%crop cross correlation central part:
if cross_correlation_algorithm==1 %fft
    [max_row,max_col] = return_max_position(cross_correlation);
    cross_correlation = cross_correlation(max_row-1:max_row+1,max_col-1:max_col+1);
    addition_to_row = max_row - (N_cross_correlation/2+1);
    addition_to_col = max_col - (N_cross_correlation/2+1);
%     cross_correlation = cross_correlation(N_cros_correlation/2:N_cros_correlation/2+2,N_cros_correlation/2:N_cros_correlation/2+2);
elseif cross_correlation_algorithm==2 %regular full
%     cross_correlation = cross_correlation(new_N/2-1:new_N/2+1,new_N/2-1:new_N/2+1);

    [max_row,max_col] = return_max_position(cross_correlation);
    cross_correlation = cross_correlation(max_row-1:max_row+1,max_col-1:max_col+1);
    addition_to_row = max_row - (N_cross_correlation/2);
    addition_to_col = max_col - (N_cross_correlation/2);
%     cross_correlation = cross_correlation(N_cros_correlation/2-upsampling_cross_correlation:N_cros_correlation/2+2-upsampling_cross_correlation,N_cros_correlation/2-upsampling_cross_correlation:N_cros_correlation/2+2-upsampling_cross_correlation);
elseif cross_correlation_algorithm==3
    cross_correlation = cross_correlation(max_row-1:max_row+1,max_col-1:max_col+1);
    addition_to_row = max_row - (calculation_count+1)/2;
    addition_to_col = max_col - (calculation_count+1)/2;
end



% shift_algorithm=4;
output_counter=1;
if ~isempty(find(shift_algorithm==1)) %regular parabola
    %get fitting points for x and fit the polynomial:
    fitting_points_x = cross_correlation(2,1:3);
    fitting_pixels_x = [-1,0,1];
    Px = polyfit(fitting_pixels_x,fitting_points_x,2);

    %get fitting points for y and fit the polynomial:
    fitting_points_y = cross_correlation(1:3,2);
    fitting_pixels_y = [-1,0,1];
    Py = polyfit(fitting_pixels_y,fitting_points_y',2);
    
%     figure(4)
%     scatter(-1:1:1,fitting_points_x,'b');
%     hold on;
%     plot(-1:0.1:1,polyval(Px,-1:0.1:1),'g');
    
    %FIND MAX OF THE POLYNOMIALS:
    shiftx=-Px(2)/(2*Px(1));
    shifty=-Py(2)/(2*Py(1));
    
    %find max: 
    x_max = polyval(Px,shiftx);
    y_max = polyval(Py,shifty);
    z_max = max(x_max,y_max);
 
    %correct for initial maximum not at the center:
    shiftx = shiftx + addition_to_col; 
    shifty = shifty + addition_to_row;
    
    %correct for proper spacing:
    shiftx=(shiftx)*cross_correlation_spacing;
    shifty=(shifty)*cross_correlation_spacing;
    if cross_correlation_algorithm==2
       shiftx=shiftx*-1;
       shifty=shifty*-1;
    elseif cross_correlation_algorithm==3
       shiftx=shiftx*-1; 
    end
    
    %update output:
    output{output_counter}=shiftx;
    output{output_counter+1}=shifty;
    output{output_counter+2}=z_max;
    output_counter=output_counter+3;
end

if ~isempty(find(shift_algorithm==2)) %regular paraboloid
    cloc=2;
    rloc=2;
    x=[1,cloc-1,cloc-1,cloc,cloc,cloc,cloc+1,cloc+1,cloc+1];
    y=[rloc-1,rloc,rloc+1,rloc-1,rloc,rloc+1,rloc-1,rloc,rloc+1];
    
    for k=1:length(x)
       cross_correlation_samples(k) = cross_correlation(x(k),y(k)); 
    end
    x=x-cloc;
    y=y-rloc;
    [coeffs] = fit_polynom_surface( x', y', cross_correlation_samples', 2 );
    shifty = (-(coeffs(2)*coeffs(5)-2*coeffs(3)*coeffs(4))/(coeffs(5)^2-4*coeffs(3)*coeffs(6)));
    shiftx = ((2*coeffs(2)*coeffs(6)-coeffs(4)*coeffs(5))/(coeffs(5)^2-4*coeffs(3)*coeffs(6)));
    
    %find max:
    z_max = evaluate_2d_polynom_surface( shiftx, shifty, coeffs );
      
    %correct for initial maximum not at the center:
    shiftx = shiftx + addition_to_col;
    shifty = shifty + addition_to_row;
    
    %correct for proper spacing:
    shiftx=shiftx*cross_correlation_spacing;
    shifty=shifty*cross_correlation_spacing;
    if cross_correlation_algorithm==2
       shiftx=shiftx*-1;
       shifty=shifty*-1;
    elseif cross_correlation_algorithm==3
       shiftx=shiftx*-1; 
    end
    
    %update output:
    output{output_counter}=shiftx;
    output{output_counter+1}=shifty;
    output{output_counter+2}=z_max;
    output_counter=output_counter+3;
end

if ~isempty(find(shift_algorithm==3)) %log parabola
    %get fitting points for x and fit the polynomial:
    fitting_points_x = log(cross_correlation(2,1:3));
    fitting_pixels_x = [-1,0,1];
    Px = polyfit(fitting_pixels_x,fitting_points_x,2);

    %get fitting points for y and fit the polynomial:
    fitting_points_y = log(cross_correlation(1:3,2));
    fitting_pixels_y = [-1,0,1];
    Py = polyfit(fitting_pixels_y,fitting_points_y',2);
    
    %FIND MAX OF THE POLYNOMIALS:
    shiftx=-Px(2)/(2*Px(1));
    shifty=-Py(2)/(2*Py(1)); 
    
   %find max:
    x_max = polyval(Px,shiftx);
    y_max = polyval(Py,shifty);
    z_max = max(x_max,y_max); 
    z_max = exp(z_max);
    
    %correct for initial maximum not at the center:
    shiftx = shiftx + addition_to_col;
    shifty = shifty + addition_to_row;
    
    %correct for proper spacing:
    shiftx=(shiftx)*cross_correlation_spacing;
    shifty=(shifty)*cross_correlation_spacing;
    if cross_correlation_algorithm==2
       shiftx=shiftx*-1;
       shifty=shifty*-1;
    elseif cross_correlation_algorithm==3
       shiftx=shiftx*-1; 
    end
    
    %update output:
    output{output_counter}=shiftx;
    output{output_counter+1}=shifty;
    output{output_counter+2}=z_max;
    output_counter=output_counter+3;
end

if ~isempty(find(shift_algorithm==4)) %log paraboloid
    cloc=2;
    rloc=2;
    x=[cloc-1,cloc-1,cloc-1,cloc,cloc,cloc,cloc+1,cloc+1,cloc+1];
    y=[rloc-1,rloc,rloc+1,rloc-1,rloc,rloc+1,rloc-1,rloc,rloc+1];

    for k=1:length(x)
%         cross_correlation_samples(k) = log(cross_correlation(x(k),y(k))); 
       cross_correlation_samples(k) = log(abs(cross_correlation(x(k),y(k)))); 
    end
    x=x-cloc; 
    y=y-rloc;
    [coeffs] = fit_polynom_surface( x', y', cross_correlation_samples', 2 );
    shifty = (-(coeffs(2)*coeffs(5)-2*coeffs(3)*coeffs(4))/(coeffs(5)^2-4*coeffs(3)*coeffs(6)));
    shiftx = ((2*coeffs(2)*coeffs(6)-coeffs(4)*coeffs(5))/(coeffs(5)^2-4*coeffs(3)*coeffs(6)));
    
    %find max:
    z_max = evaluate_2d_polynom_surface( shiftx, shifty, coeffs );
    z_max = exp(z_max);    
     
    %correct for initial maximum not at the center:
    shiftx = shiftx + addition_to_col;
    shifty = shifty + addition_to_row;
    
    %correct for proper spacing:
    shiftx=shiftx*cross_correlation_spacing; 
    shifty=shifty*cross_correlation_spacing;
    if cross_correlation_algorithm==2
       shiftx=shiftx*-1;
       shifty=shifty*-1;
    elseif cross_correlation_algorithm==3
       shiftx=shiftx*-1; 
    end
    
    %update output:
    output{output_counter}=shiftx;
    output{output_counter+1}=shifty;
    output{output_counter+2}=z_max;
    output_counter=output_counter+3;
end

if ~isempty(find(shift_algorithm==5)) %log plus one parabola
    
    %get fitting points for x and fit the polynomial:
    fitting_points_x = log(1+cross_correlation(2,1:3));
    fitting_pixels_x = [-1,0,1];
    Px = polyfit(fitting_pixels_x,fitting_points_x,2);

    %get fitting points for y and fit the polynomial:
    fitting_points_y = log(1+cross_correlation(1:3,2));
    fitting_pixels_y = [-1,0,1];
    Py = polyfit(fitting_pixels_y,fitting_points_y',2);
    
    %FIND MAX OF THE POLYNOMIALS:
    shiftx=-Px(2)/(2*Px(1));
    shifty=-Py(2)/(2*Py(1)); 
    
    %find max:
    x_max = polyval(Px,shiftx);
    y_max = polyval(Py,shifty);
    z_max = max(x_max,y_max);
    z_max = exp(z_max-1);
    
    %correct for initial maximum not at the center:
    shiftx = shiftx + addition_to_col;
    shifty = shifty + addition_to_row;
    
    %correct for proper spacing:
    shiftx=(shiftx)*cross_correlation_spacing;
    shifty=(shifty)*cross_correlation_spacing;
    if cross_correlation_algorithm==2
       shiftx=shiftx*-1;
       shifty=shifty*-1;
    elseif cross_correlation_algorithm==3
       shiftx=shiftx*-1; 
    end
    
    %update output:
    output{output_counter}=shiftx;
    output{output_counter+1}=shifty;
    output{output_counter+2}=z_max;
    output_counter=output_counter+3;
end

if ~isempty(find(shift_algorithm==6)) %log plus one paraboloid
    cloc=2;
    rloc=2;
    x=[cloc-1,cloc-1,cloc-1,cloc,cloc,cloc,cloc+1,cloc+1,cloc+1];
    y=[rloc-1,rloc,rloc+1,rloc-1,rloc,rloc+1,rloc-1,rloc,rloc+1];

    for k=1:length(x)
       cross_correlation_samples(k) = log(1+cross_correlation(x(k),y(k))); 
    end
    x=x-cloc; 
    y=y-rloc;
    [coeffs] = fit_polynom_surface( x', y', cross_correlation_samples', 2 );
    shifty = (-(coeffs(2)*coeffs(5)-2*coeffs(3)*coeffs(4))/(coeffs(5)^2-4*coeffs(3)*coeffs(6)));
    shiftx = ((2*coeffs(2)*coeffs(6)-coeffs(4)*coeffs(5))/(coeffs(5)^2-4*coeffs(3)*coeffs(6)));
    
    %find max:
    z_max = evaluate_2d_polynom_surface( shiftx, shifty, coeffs );
    z_max = exp(z_max-1);
    
    %correct for initial maximum not at the center:
    shiftx = shiftx + addition_to_col;
    shifty = shifty + addition_to_row;
    
    %correct for proper spacing:
    shiftx=shiftx*cross_correlation_spacing;
    shifty=shifty*cross_correlation_spacing;
    if cross_correlation_algorithm==2
       shiftx=shiftx*-1;
       shifty=shifty*-1;
    elseif cross_correlation_algorithm==3
       shiftx=shiftx*-1; 
    end  

    %update output:
    output{output_counter}=shiftx;
    output{output_counter+1}=shifty;
    output{output_counter+2}=z_max;
    output_counter=output_counter+3;
end 

if ~isempty(find(shift_algorithm==7)) %lagrange surface
    cloc=2;
    rloc=2;
    x=[cloc-1,cloc-1,cloc-1,cloc,cloc,cloc,cloc+1,cloc+1,cloc+1];
    y=[rloc-1,rloc,rloc+1,rloc-1,rloc,rloc+1,rloc-1,rloc,rloc+1];
    
    for k=1:length(x)
       cross_correlation_samples(k) = cross_correlation(x(k),y(k)); 
    end
    x=x-cloc;
    y=y-rloc;
    [coeffs] = fit_lagrange_surface( x', y', cross_correlation_samples');
    
    %instead of: A+By+Cy^2+Dx+Exy+Fxy^2+Gx^2+Hx^2y+Ix^2y^2
    %we want to use: Ax^2y^2+Bx^2y+Cx^2+Dxy^2+Exy+Fx+Gy^2+Hy+I
    coeffs = flipud(coeffs); 
    
    %SOLVE gradient=0 to find maximum:
    A=coeffs(1);
    B=coeffs(2);
    C=coeffs(3);
    D=coeffs(4);
    E=coeffs(5);
    F=coeffs(6);
    G=coeffs(7); 
    H=coeffs(8);
    I=coeffs(9);
    str1=strcat('2*',num2str(A),'*x*y^2+2*',num2str(B),'*x*y+2*',num2str(C),'*x+',num2str(D),'*y^2+',num2str(E),'*y+',num2str(F),'=0');
    str2=strcat('2*',num2str(A),'*x^2*y+',num2str(B),'*x^2+2*',num2str(D),'*x*y+',num2str(E),'*x+2*',num2str(G),'*y+',num2str(H),'=0');
    [max_x,max_y]=solve(str1,str2);
    
    max_x=subs(max_x);
    max_y=subs(max_y);
    
    where_only_real = (imag(max_x)==0 & imag(max_y)==0);
    max_x=max_x(where_only_real==1);
    max_y=max_y(where_only_real==1);
    a=max_x.^2+max_y.^2;
    shiftx=max_x(a==min(a));
    shifty=max_y(a==min(a));
    
    %find max (MISSING FOR NOW- REPLACE THIS):
    z_max=0;
     
    %correct for initial maximum not at the center:
    shiftx = shiftx + addition_to_col;
    shifty = shifty + addition_to_row;
    
    shiftx=shiftx*-1;
    shifty=shifty*-1;
    
    %correct for proper spacing:
    shiftx=shiftx*cross_correlation_spacing;
    shifty=shifty*cross_correlation_spacing;
    if cross_correlation_algorithm==2
       shiftx=shiftx*-1;
       shifty=shifty*-1;
    elseif cross_correlation_algorithm==3
       shiftx=shiftx*-1; 
    end 
    
    %update output:
    output{output_counter}=shiftx;
    output{output_counter+1}=shifty;
    output{output_counter+2}=z_max;
    output_counter=output_counter+3;
    
end
    
    
    
    
    
    
    
    





