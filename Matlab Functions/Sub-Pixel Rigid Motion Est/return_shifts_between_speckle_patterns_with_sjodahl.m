% function [output] = ...
%     return_shifts_between_speckle_patterns_with_sjodahl(...
%     mat1,...
%     mat2,...
%     spacing,...
%     cross_correlation_algorithm,...
%     shift_algorithm,...
%     upsampling_images,...
%     upsampling_cross_correlation,...
%     flag_check_only_ROI_3,...
%     number_of_sjodahl_sub_shifts,...
%     flag_check_original_using_fourier,...
%     flag_what_to_do_after_shift,...
%     flag_use_fourier_for_final_registration)

% output:final_shiftx,final_shifty,final_z_max,final_shiftx_fourier,final_shifty_fourier,final_z_max_fourier,initial_shiftx_fit,initial_shifty_fit,initial_z_max_fit,initial_shiftx_fourier,initial_shifty_fourier,initial_z_max_fourier

%return_shifts_between_speckle_patterns returns the movement of mat2 with
%respect to mat1.
%if mat2 is shifted 1 pixel RIGHT the function returns shiftx=1
%if mat2 is shifted 1 pixel DOWN the function returns shifty=1;
%in other words: the shift is defined with respect to positive shift of row
%and column, not regular xy.

N=512;
mat1=create_speckles_of_certain_size_in_pixels(10,N,1,0);
mat2=shift_matrix(mat1,1,0.6,-0.8);
mat1=abs(mat1).^2;
mat2=abs(mat2).^2;
spacing=1;
cross_correlation_algorithm=3;
shift_algorithm=[1];
upsampling_images=1;
upsampling_cross_correlation=1;
flag_check_only_ROI_3=0; 
ROI=100;
mat1=mat1(1:ROI,1:ROI);
mat2=mat2(1:ROI,1:ROI);
number_of_sjodahl_sub_shifts=2;
flag_check_original_using_fourier=1;
flag_what_to_do_after_shift=1; %1 for chipping off edges, 2 for zoom in
flag_check_final_using_fourier=2;
[output1] = return_shifts_between_speckle_patterns(mat1,mat2,spacing,cross_correlation_algorithm,shift_algorithm,upsampling_images,upsampling_cross_correlation,flag_check_only_ROI_3);
   
%initialize shift tracking variables:
output_counter=6;
fit_counter = 3;
if flag_check_original_using_fourier==1
   fit_counter = 6;
end

if flag_check_final_using_fourier==1
   output_counter=output_counter+3; 
elseif flag_check_final_using_fourier==2
   output_counter=output_counter+6; 
end
%initialize an output variable with a number of array cells equal to
%output_counter according to what i want to get (only surface fit, only
%fourier, or both):
output = zeros(output_counter,1);

for k=1:number_of_sjodahl_sub_shifts
    %find shift under current conditions:
    [output1] = return_shifts_between_speckle_patterns(mat1,mat2,spacing,cross_correlation_algorithm,shift_algorithm,upsampling_images,upsampling_cross_correlation,flag_check_only_ROI_3);
    shiftx = output1{1};
    shifty = output1{2};
    z_max = output1{3};
    return_shifts_with_sub_pixel_shift
    if k==1
       %initial values:
       output(1) = shiftx;
       output(2) = shifty;
       output(3) = z_max;
       if flag_check_original_using_fourier==1
           %if so requested find shift using fourier without doing anything
           %like Dan's algorithm:
           [initial_shiftx_fourier,initial_shifty_fourier,initial_z_max_fourier] = return_shifts_with_fourier_sampling(mat1,mat2,1,1000);
           [initial_shiftx_fourier1,initial_shifty_fourier1,initial_z_max_fourier1] = return_shifts_with_fourier_sampling(mat1,mat1,1,100);
           [initial_shiftx_fourier2,initial_shifty_fourier2,initial_z_max_fourier2] = return_shifts_with_fourier_sampling(mat2,mat2,1,100);
           output(4)=initial_shiftx_fourier;
           output(5)=initial_shifty_fourier;
           output(6)=abs(initial_z_max_fourier)/sqrt(abs(initial_z_max_fourier1)*abs(initial_z_max_fourier2));
       end
    end
     
    %update final fit shiftx:
    if k<number_of_sjodahl_sub_shifts
        output(fit_counter+1)=output(fit_counter+1)+shiftx;
        output(fit_counter+2)=output(fit_counter+2)+shifty;
        output(fit_counter+3)=z_max;
    end
       
    if number_of_sjodahl_sub_shifts>1
        
        %chop off decorrelated areas column areas:
        if shifty<-0.5
            mat1 = mat1(1+round(abs(shifty)):end,1:end);
            mat2 = mat2(1:end-round(abs(shifty)),1:end);
        elseif shifty>0.5 
            mat2 = mat2(1+round(abs(shifty)):end,1:end);
            mat1 = mat1(1:end-round(abs(shifty)),1:end); 
        end 

        %chop off decorrelated areas column areas:
        if shiftx>0.5
            mat2 = mat2(1:end,1+round(shiftx):end);
            mat1 = mat1(1:end,1:end-round(shiftx));
        elseif shiftx<-0.5
            mat1 = mat1(1:end,1+round(abs(shiftx)):end);
            mat2 = mat2(1:end,1:end-round(abs(shiftx)));
        end   
    
        %sub pixel shift the matrix:
        %here i shift using fourier transform, it might be advisable to use
        %interp2 instead...
        %i should also decide whether to first minimially chop off ROI then
        %shift or shift then maximally chop off ROI
        %also maybe i should use my interp2_ft function to zoom in to the image
        %in order to eventually completely disregard decorrelation regions
        n2=size(mat1,1);
        m2=size(mat2,2);
        sub_pixel_shift_x = (shiftx-round(shiftx));
        sub_pixel_shift_y = (shifty-round(shifty));
        mat2 = abs(shift_matrix(mat2,1,-sub_pixel_shift_x,-sub_pixel_shift_y));

        %decide what to do after shift:
        if flag_what_to_do_after_shift==1 || k==1
            %chop off edges:
            mat1 = mat1(2:end-1,2:end-1);
            mat2 = mat2(2:end-1,2:end-1);
        elseif flag_what_to_do_after_shift==2
            %zoom in to avoid decorrelation caused by the edges after the sub-shift:
            mat1 = abs(interp2_ft(mat1,n2/(n2-1),0,0,1,1));
            mat2 = abs(interp2_ft(mat2,n2/(n2-1),0,0,1,1));
        end    

    end 
     
    
    %if at final run of loop:
    if k==number_of_sjodahl_sub_shifts
        if flag_check_final_using_fourier==0
            %use only surface fit:
            [output1] = return_shifts_between_speckle_patterns(mat1,mat2,spacing,cross_correlation_algorithm,shift_algorithm,upsampling_images,upsampling_cross_correlation,flag_check_only_ROI_3);
            %update final fit shiftx:
            shiftx = output1{1};
            shifty = output1{2};
            z_max = output1{3};
            output(fit_counter+1)=output(fit_counter+1)+shiftx;
            output(fit_counter+2)=output(fit_counter+2)+shifty;
            output(fit_counter+3)=z_max;
        elseif flag_check_final_using_fourier==1
            %use only fourier:
            [shiftx,shifty,z_max] = return_shifts_with_fourier_sampling(mat1,mat2,1,2000);
            [initial_shiftx_fourier1,initial_shifty_fourier1,initial_z_max_fourier1] = return_shifts_with_fourier_sampling(mat1,mat1,1,100);
            [initial_shiftx_fourier2,initial_shifty_fourier2,initial_z_max_fourier2] = return_shifts_with_fourier_sampling(mat2,mat2,1,100);
            %update final fit shiftx:
            output(fit_counter+1)=shiftx;
            output(fit_counter+2)=shifty;
            output(fit_counter+3)=abs(z_max)/sqrt(abs(initial_z_max_fourier1)*abs(initial_z_max_fourier2));
        elseif flag_check_final_using_fourier==2
            %use both surface fit and fourier:
            [output1] = return_shifts_between_speckle_patterns(mat1,mat2,spacing,cross_correlation_algorithm,shift_algorithm,upsampling_images,upsampling_cross_correlation,flag_check_only_ROI_3);
            [shiftx_fourier,shifty_fourier,z_max_fourier] = return_shifts_with_fourier_sampling(mat1,mat2,1,2000);
            [initial_shiftx_fourier1,initial_shifty_fourier1,initial_z_max_fourier1] = return_shifts_with_fourier_sampling(mat1,mat1,1,100);
            [initial_shiftx_fourier2,initial_shifty_fourier2,initial_z_max_fourier2] = return_shifts_with_fourier_sampling(mat2,mat2,1,100);
            shiftx = output1{1};
            shifty = output1{2};
            z_max = output1{3};
            output(fit_counter+1)=output(fit_counter+1)+shiftx;
            output(fit_counter+2)=output(fit_counter+2)+shifty;
            output(fit_counter+3)=z_max;
            output(fit_counter+4)=output(fit_counter+1)+shiftx_fourier;
            output(fit_counter+5)=output(fit_counter+2)+shifty_fourier;
            output(fit_counter+6)=abs(z_max_fourier)/sqrt(abs(initial_z_max_fourier1)*abs(initial_z_max_fourier2));
        end 
    end 
end 
1; 
