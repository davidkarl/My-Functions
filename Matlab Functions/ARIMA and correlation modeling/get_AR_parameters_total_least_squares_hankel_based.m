function [AR_parameters] = get_AR_parameters_total_least_squares_hankel_based(input_signal,AR_model_order)
% Hankel Total Least Squares based AR model parameter estimation
%
% 'x' containts measurements corrupted by white noise
% on output 'a' contains the coefficients of an AR system of the order 'n'
% generating output 'y', while (x-y) is minimised in Euclidean sense
% y(i)=a(1)*y(i-1)+a(2)*y(i-2)+...+a(n)*y(i-n)
% a=arg min sum((y(i)-x(i))^2)
%
% Usage: a=htls(x,n) ;

input_signal = input_signal(:);
input_signal_length = length(input_signal);

%get Hankel matrix number of rows:
hankel_matrix_number_of_rows = floor(input_signal_length/2);
%get Hankel matrix number of columns:
hankel_matrix_number_of_columns = input_signal_length + 1 - hankel_matrix_number_of_rows;

%construct the matrix:
X_hankel_matrix = zeros(hankel_matrix_number_of_rows,hankel_matrix_number_of_columns);
for i = 1:hankel_matrix_number_of_rows
    X_hankel_matrix(i,:) = input_signal( i : i+hankel_matrix_number_of_columns-1 )';
end

[u , s , v] = svd(X_hankel_matrix,0);
z = total_least_squares( u(1:hankel_matrix_number_of_rows-1 , 1:AR_model_order) , ...
         u(2:hankel_matrix_number_of_rows , 1:AR_model_order) );
% z=u(1:l-1,1:n)\u(2:l,1:n) ;
e = eig(z);
for i=1:AR_model_order
    if abs(e(i))>1
        e(i) = 1;
    end
end 
  
AR_parameters = poly(e);
AR_parameters = -AR_parameters(2:AR_model_order+1) ;