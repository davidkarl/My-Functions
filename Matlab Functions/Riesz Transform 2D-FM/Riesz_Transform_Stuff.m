%Generate a circle image
img = zeros(512, 512);
img(:) = 128;
rad = 180;
for i = size(img, 1)/2 - rad : size(img,1)/2 + rad
    for j = size(img, 2)/2 - rad : size(img,2)/2 + rad
        deltaX = abs(size(img, 1)/2 - i);
        deltaY = abs(size(img, 2)/2 - j);
        if (sqrt(deltaX^2+deltaY^2) <= rad)
           img(i, j) = 255;
        end
    end
end

%build riesz pyramid
[pyr, pind] = buildNewPyr(img);

%extract band2 from pyramid (no orientation information yet)
I = pyrBand(pyr,pind,3);

%convolve band2 with approximate riesz filter for first quadrature pair
%element 
R1 = conv2(I, [0.5, 0, -0.5], 'same');

%convolve band2 with approximate riesz filter (rotated by 90°) for second
%quadrature pair element
R2 = conv2(I, [0.5, 0, -0.5]', 'same');

% show the resulting image containing orientation information!
% imshow(band2_r2, []);

%To extract the phase, we have to steer the pyramid to its dominant local
%orientation. Orientation is calculated as atan(R2/R1)
theta = atan(R2./R1);
theta(isnan(theta) | isinf(theta)) = 0;
%imshow(theta, []);

% create quadrature pair
Q = zeros(size(theta, 1), size(theta, 2));
 
for i = 1:size(theta, 1)
    for j = 1:size(theta, 1)
        if theta(i, j) ~= 0
            %create rotation matrix
            rot_mat = ...
                [cos(theta(i, j)), sin(theta(i, j));...
                -sin(theta(i, j)) cos(theta(i, j))];

            %steer to dominant local orientation(theta) and set Q
            resultPair = rot_mat*[R1(i, j), R2(i,j)]';
            Q(i,j) = resultPair(1);
        end 
    end
end

% create amplitude and phase image
A = abs(complex(I, Q));
Phi = angle(complex(I, Q));