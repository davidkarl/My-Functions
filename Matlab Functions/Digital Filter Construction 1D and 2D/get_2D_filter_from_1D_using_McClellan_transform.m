function h = get_2D_filter_from_1D_using_McClellan_transform(filter_1D,transform_matrix)
% MCTRANS McClellan transformation
%   H = mctrans(B,T) produces the 2-D FIR filter H that
%   corresponds to the 1-D FIR filter B using the transform T.


% Convert the 1-D filter b to SUM_n a(n) cos(wn) form
filter_1D_center = (length(filter_1D)-1)/2;
filter_1D = rot90(fftshift(rot90(filter_1D,2)),2); % Inverse fftshift
a = [filter_1D(1) 2*filter_1D(2:filter_1D_center+1)];

transform_matrix_center = floor((size(transform_matrix)-1)/2);

%Use Chebyshev polynomials to compute h:
P0 = 1; 
P1 = transform_matrix;
h = a(2)*P1; 
rows = transform_matrix_center(1)+1; cols = transform_matrix_center(2)+1;
h(rows,cols) = h(rows,cols)+a(1)*P0;
for i = 3:filter_1D_center+1,
    P2 = 2*conv2(transform_matrix,P1);
    rows = rows + transform_matrix_center(1); 
    cols = cols + transform_matrix_center(2);
    P2(rows,cols) = P2(rows,cols) - P0;
    rows = transform_matrix_center(1) + [1:size(P1,1)];
    cols = transform_matrix_center(2) + [1:size(P1,2)];
    hh = h;
    h = a(i)*P2; 
    h(rows,cols) = h(rows,cols) + hh;
    P0 = P1;
    P1 = P2;
end
h = rot90(h,2); % Rotate for use with filter2