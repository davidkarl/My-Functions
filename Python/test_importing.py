##Test import
#import my_linspace
#from my_linspace import my_linspace
#print(my_linspace(1,1,6));

mat_in = speckles;
number_of_center_pixels = 60;

mat_in_shape = shape(mat_in);
mat_in_rows = mat_in_shape[0];
mat_in_cols = mat_in_shape[1];
mat_in_rows_excess = mat_in_rows - number_of_center_pixels;
mat_in_cols_excess = mat_in_cols - number_of_center_pixels;
blabla = mat_in[int(start + mat_in_rows_excess/2) : int(end - mat_in_rows_excess/2), \
              int(start + mat_in_cols_excess/2) : int(end-mat_in_cols_excess/2)];

