%dsp book generating an exact gaussian AR
%polynomial generation

%parameters:
process_noise_variance=2;
number_of_distinct_poles=5;

%generate number_of_distinct_poles modulai and phases:
rho=0.45*rand(1,number_of_distinct_poles)+0.5;
phi=pi*rand(1,number_of_distinct_poles)/4;

%use complex conjugate symmetry and calculate the complex poles:
rac=rho.*exp(1*phi);
rac=[rac,conj(rac)];

%create a polynomial based on all the pole pairs generated above:
coeff=real(poly(rac));
polynomial_order=length(coeff)-1;
 
%calculation of the covariance R(k) using an alternative formulation of the
%yule-walker equations where {ai} and process noise variance are known and we're looking for R(k):
first_column = [coeff(polynomial_order+1);zeros(polynomial_order,1)];
first_row = coeff(polynomial_order+1:-1:1);
A1=toeplitz( first_column, first_row );
A1=A1(:,polynomial_order+1:-1:1);
A2=toeplitz(coeff,[coeff(1);zeros(polynomial_order,1)]);
rx=(A1+A2)\[process_noise_variance;zeros(polynomial_order,1)]; 
rx(1)=rx(1)*2; %in the inverse formulation of the YW equations the first term is R(0)/2 so we correct for it
Rcov=toeplitz(rx);

%calculate square root of the covarance matrix to transform the signal into white noise
%using the fact that the covariance matrix of a transformed process with transformation matrix
%M (y=Mx) can be written as Ry=MRxM', so if we want to whitten the signal
%we need M to be sqrt(Rx). conversely, what we do he is calculate sqrtm(R)
%to change a later variable W0 which is white into a process described by
%Rcov above, built from {ai}:
MatM=sqrtm(Rcov(1:polynomial_order,1:polynomial_order));
 

%generating the first Ncoeff-1 values of X(k)
W0=randn(polynomial_order,1); 
X0=MatM*W0; 
Z0=filtricII(1,coeff,0,X0);
T=200; 
W=sqrt(process_noise_variance)*randn(T,1);
Xf=filter(1,coeff,W,Z0); 
LXf=length(Xf);
Xf=Xf-mean(Xf); 
R=2*polynomial_order;
covX=zeros(R,1);

for rr=1:R
    covX(rr)=Xf(rr:T)'*Xf(1:T-rr+1)/T; 
end

subplot(2,2,1); mycirc=exp(2*pi*1i*[0:100]/100); plot(mycirc);
hold on; plot(rac,'x'); hold off; axis('square'); grid;

subplot(2,2,2); plot(Xf); grid
subplot(2,2,3); stem(covX/covX(1));
subplot(2,2,4); plot(Xf(1:LXf-1),Xf(2:LXf),'.'); grid






