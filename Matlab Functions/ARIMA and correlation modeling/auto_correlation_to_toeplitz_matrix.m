function [Toeplitz_auto_correlation_matrix]=auto_correlation_to_toeplitz_matrix(auto_correlation_sequence)
% Given autocorrelation coefficients, build the Toeplitz matrix
% Usage: R=c2R(c) ;

auto_correlation_sequence=auto_correlation_sequence(:)' ;
n=length(auto_correlation_sequence) ;
cc=[auto_correlation_sequence(n:-1:2) auto_correlation_sequence];
for i=1:n,
  Toeplitz_auto_correlation_matrix(i,:)=cc(n-i+1:2*n-i) ;
end ;  
