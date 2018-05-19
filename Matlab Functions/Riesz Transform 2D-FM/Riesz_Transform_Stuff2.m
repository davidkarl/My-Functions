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







% I have got a functioning implementation of this algorithm. 
% Here are the steps I took to successfully motion-magnify a video using this method.
% 
% These steps should be applied to each channel of a video sequence that you 
% (I have tried it for RGB video, you could probably get away with doing it for just luminance, in a YUV video).
% 
% Create an image pyramid of each frame. 
% The original paper has a recommended pyramid structure to allow greater magnification values, 
% although it works fairly well with a Laplacian pyramid.

% For each pyramid level of each video channel, calculate the Riesz transform 
% (see The Riesz transform and simultaneous representations of phase, energy and orientation in spatial 
% vision for an overview of the transform, and see the paper in the original question for an 
% efficient approximate implementation).

% Using the Riesz transform, calculate the local amplitude, orientation and phase for each pixel of
% each pyramid level of each video frame. The following Matlab code will calculate the local orientation, 
% phase and amplitude of a (double format) image using the approximate Riesz transform:


function [orientation, phase, amplitude] = riesz(image)

[imHeight, imWidth] = size(image);

%approx riesz, from Riesz Pyramids for Fast Phase-Based Video Magnification
 
dxkernel = zeros(size(image));
dxkernel(1, 2)=-0.5;
dxkernel(1,imWidth) = 0.5;


dykernel = zeros(size(image));
dykernel(2, 1) = -0.5;
dykernel(imHeight, 1) = 0.5;

R1 = ifft2(fft2(image) .* fft2(dxkernel));
R2 = ifft2(fft2(image) .* fft2(dykernel));


orientation = zeros(imHeight, imWidth);
phase = zeros(imHeight, imWidth);

orientation = (atan2(-R2, R1));

phase = ((unwrap(atan2(sqrt(R1.^2 + R2.^2) , image))));

amplitude = sqrt(image.^2 + R1.^2 + R2.^2);

end  


% For each pyramid level, temporally filter the phase values of each pixel using a 
% bandpass filter set to a frequency appropriate for the motion that you wish to magnify. 
% Note that this removes the DC component from the phase value.
% Calculate the amplified phase value by amplifiedPhase = phase + (requiredGain * filteredPhase);
% Use the amplified phase to calculate new pixel values for each pyramid level by
% amplifiedSequence = amplitude .* cos(amplifiedPhase);
% Collapse the pyramids to generate the new, amplified video channel.
% Recombine your amplified channels into a new video frame.



%!!!!!!!!!!!!!!!!!!
%  worked out a very similar solution (I actually stumbled upon the same document you linked). 
%  The original document states that the phase can't naively be filtered but rather the
%  quantities "?cos(?)" and "?sin(?)". 
%  Did you, by any chance, try that? In my expermients it lead to a significantly increased amount of artifacts, 
%  suggesting I overlooked something.
 
 


%%!!!!!!!!!!!!!!!!!!!!!!!!11
%  have fully implemented the methodology of Riesz Pyramids for fast phase based video motion magnification. I felt the papers did not clearly describe the appropriate steps required to correctly filter phase. It is important to realise that mulitple mathematically correct expressions for phase and orientation may in fact not be suitable using MATLAB's acos(), asin() and atan() functions. Here is my implementation:

% R1, R2 are Riesz transforms of the image I and Q is the Quadrature pair

Q = sqrt((R1.^2) + (R2.^2));

phase = atan2(Q,I);
% The phase should be wrapped to be between -pi and +pi, i.e. if phase is greater than +pi, phase = phase - 2*pi, if phase is smaller than -pi,

phase = phase + 2*pi.

amplitude = sqrt((I.^2) + (R1.^2) + (R2.^2));
% Furthermore, it is essential that the change in phase between successive frames is filtered, not the phase directly.

phase_diff = phase(t+1) - phase(t);
% This quantity "phase_diff" is temporally filtered, and amplified by multiplication with an amplification factor. The filtered, amplified, variation in phase is then used to phase shift the input.

% magnified output = amplitude.*cos(phase_diff_filtered_amplified + original phase).



