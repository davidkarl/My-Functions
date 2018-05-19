function [shiftx,shifty] = return_shifts_with_sub_pixel_shift(mat1,mat2,spacing,accuracy,maximum_number_of_loops)

% accuracy=3/1000;
% maximum_number_of_loops=3;

%just initialize to some value so that total_shift>accuracy:
shiftx=0;
shifty=0;

%initialize loop counter to avoid an infinite loop:
counter=1;

%at first use the predictable fourier upsampling algorithm:
while (sqrt(shiftx^2+shifty^2)>accuracy && counter<=maximum_number_of_loops) || counter==1 
[col_shift,row_shift,CCmax] = return_shifts_with_fourier_sampling(mat1,mat2,1000);
mat2 = shift_matrix(mat2,1,-col_shift,-row_shift);

%chip off uncorrelated areas:
mat1 = mat1(1+ceil(abs(col_shift)):end-ceil(abs(col_shift)),1+ceil(abs(row_shift)):end-ceil(abs(row_shift))-1);
mat2 = mat2(1+ceil(abs(col_shift)):end-ceil(abs(col_shift)),1+ceil(abs(row_shift)):end-ceil(abs(row_shift))-1);

%update extrapolated shifts:
shiftx = shiftx + col_shift;
shifty = shifty + row_shift;

%update counter:
counter = counter+1;
end

shiftx=shiftx*spacing;
shifty=shifty*spacing;



