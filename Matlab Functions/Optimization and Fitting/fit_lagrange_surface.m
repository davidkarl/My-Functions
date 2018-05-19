function [coeffs] = fit_lagrange_surface(x,y,z)

[sizexR, sizexC] = size(x);
[sizeyR, sizeyC] = size(y);
[sizezR, sizezC] = size(z);

if (sizexC ~= 1) || (sizeyC ~= 1) || (sizezC ~= 1)
    fprintf( 'Inputs of fit2dPolySVD must be column vectors' );
    return;
end

if (sizeyR ~= sizexR) || (sizezR ~= sizexR)
    fprintf( 'Inputs vectors of fit2dPolySVD must be the same length' );
    return;
end


numVals = sizexR;

% scale to prevent precision problems
scalex = 1.0/max(abs(x));
scaley = 1.0/max(abs(y));
scalez = 1.0/max(abs(z));
xs = x .* scalex;
ys = y .* scaley;
zs = z .* scalez;


% number of combinations of coefficients in resulting polynomial
numCoeffs = 9;

% Form array to process with SVD
A = zeros(numVals, numCoeffs);

order=2; %lagrangian surface
column = 1;
for xpower = 0:order
    for ypower = 0:order
        A(:,column) = xs.^xpower .* ys.^ypower;
        column = column + 1;
    end
end

% Perform SVD
[u, s, v] = svd(A);

% pseudo-inverse of diagonal matrix s
sigma = eps^(1/order); % minimum value considered non-zero
qqs = diag(s);
qqs(abs(qqs)>=sigma) = 1./qqs(abs(qqs)>=sigma);
qqs(abs(qqs)<sigma)=0;
qqs = diag(qqs);
if numVals > numCoeffs
    qqs(numVals,1)=0; % add empty rows
end

% calculate solution
coeffs = v*qqs'*u'*zs; 


% scale the coefficients so they are correct for the unscaled data
column = 1;
for xpower = 0:order
    for ypower = 0:(order-xpower)
        coeffs(column) = coeffs(column) * scalex^xpower * scaley^ypower / scalez;
        column = column + 1;
    end
end








