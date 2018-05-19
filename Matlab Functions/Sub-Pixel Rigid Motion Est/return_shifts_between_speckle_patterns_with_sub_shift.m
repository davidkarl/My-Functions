function [output] = ...
    return_shifts_between_speckle_patterns_with_sub_shift(...
    mat1,...
    mat2,...
    spacing,...
    cross_correlation_algorithm,...
    shift_algorithm,...
    upsampling_images,...
    upsampling_cross_correlation,...
    flag_check_only_ROI_3,...
    number_of_sjodahl_sub_shifts,...
    flag_what_to_do_after_shift,...
    flag_use_fit_or_fourier,...
    fourier_upsampling_accuracy)

% output:
% {
% first_shiftx
% first_shifty
% first_z_max
% second_shiftx
% second_shifty
% second_z_max
% ....
% }

%if number_of_sjodahl_sub_shifts==0 this simply gives the regular result 

%return_shifts_between_speckle_patterns returns the movement of mat2 with
%respect to mat1.
%if mat2 is shifted 1 pixel RIGHT the function returns shiftx=1
%if mat2 is shifted 1 pixel DOWN the function returns shifty=1;
%in other words: the shift is defined with respect to positive shift of row
%and column, not regular xy.

% N=512;
% mat1=create_speckles_of_certain_size_in_pixels(10,N,1,0);
% mat2=shift_matrix(mat1,1,0,0);
% mat1=abs(mat1).^2;
% mat2=abs(mat2).^2;
% spacing=1;
% cross_correlation_algorithm=3;
% shift_algorithm=[4];
% upsampling_images=1;
% upsampling_cross_correlation=1;
% flag_check_only_ROI_3=0; 
% ROI=100;
% mat1=mat1(1:ROI,1:ROI);
% mat2=mat2(1:ROI,1:ROI);
% number_of_sjodahl_sub_shifts=1;
% flag_use_fit_or_fourier=2; %1 for fit, 2 for fourier
% flag_what_to_do_after_shift=1; %1 for chipping off edges, 2 for zoom in
% fourier_upsampling_accuracy=2000;
% [output1] = return_shifts_between_speckle_patterns(mat1,mat2,spacing,cross_correlation_algorithm,shift_algorithm,upsampling_images,upsampling_cross_correlation,flag_check_only_ROI_3);
    
%initialize shift tracking variables:
output = zeros(3*(number_of_sjodahl_sub_shifts+1),1);
output_counter=1;
for k=0:number_of_sjodahl_sub_shifts
    
    if flag_use_fit_or_fourier==1
        %find shift under current conditions with surface fit:
        [output1] = return_shifts_between_speckle_patterns(mat1,mat2,spacing,cross_correlation_algorithm,shift_algorithm,upsampling_images,upsampling_cross_correlation,flag_check_only_ROI_3);
        shiftx = output1{1};
        shifty = output1{2};
        z_max = output1{3};
        if k==0
            output(1) = shiftx;
            output(2) = shifty;
            output(3) = z_max;
            output_counter=output_counter+3;
        else
            output(output_counter)=output(output_counter-3)+shiftx;
            output(output_counter+1)=output(output_counter-2)+shifty;
            output(output_counter+2)=z_max;
            output_counter=output_counter+3;
        end
    elseif flag_use_fit_or_fourier==2
        %find shift using fourier upsampling:
        [shiftx,shifty,z_max] = return_shifts_with_fourier_sampling(mat1,mat2,1,fourier_upsampling_accuracy);
        if k==0
            output(1)=shiftx;
            output(2)=shifty;
            output(3)=z_max;
            output_counter=output_counter+3;
        else
            output(output_counter)=output(output_counter-3)+shiftx;
            output(output_counter+1)=output(output_counter-2)+shifty;
            output(output_counter+2)=z_max;
            output_counter=output_counter+3;
        end
    end
    
       
    %chop off edges or shift or zoom:
    if number_of_sjodahl_sub_shifts>=1 && k<number_of_sjodahl_sub_shifts
        
        if flag_what_to_do_after_shift==1
            
            %chip off uncorrelated areas:
            mat2 = abs(shift_matrix(mat2,1,-shiftx,-shifty));
            if shifty<0 %mat2 is pushed up with respect to mat1
                mat1 = mat1(1+ceil(abs(shifty)):end,1:end);
                mat2 = mat2(1+ceil(abs(shifty)):end,1:end);
            elseif shifty>0 %mat2 is pushed down with respect to mat1
                mat2 = mat2(1:end-ceil(shifty),1:end);
                mat1 = mat1(1:end-ceil(shifty),1:end); 
            end 

            %chop off decorrelated areas column areas:
            if shiftx>0
                mat1 = mat1(1:end,1:end-ceil(shiftx));
                mat2 = mat2(1:end,1:end-ceil(shiftx));
            elseif shiftx<0
                mat2 = mat2(1:end,1+ceil(abs(shiftx)):end);
                mat1 = mat1(1:end,1+ceil(abs(shiftx)):end);
            end   
        
        elseif flag_what_to_do_after_shift==2
            n2=size(mat1,1);
            m2=size(mat2,2);
            %zoom in to avoid decorrelation caused by the edges after the sub-shift:
            mat1 = abs(interp2_ft(mat1,n2/(n2-max(ceil(abs(shiftx)),ceil(abs(shifty)))),0,0,1,1));
            mat2 = abs(interp2_ft(mat2,m2/(m2-max(ceil(abs(shiftx)),ceil(abs(shifty)))),0,0,1,1));
        end
    end 
     

end 
1;   
