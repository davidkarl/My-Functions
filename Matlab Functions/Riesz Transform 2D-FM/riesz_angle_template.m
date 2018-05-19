function [steering_angles_matrix,steered_riesz_matrix_3D] = ...
                   riesz_angle_template(riesz_matrix_3D, ...
                                        linear_combination_coefficients_of_riesz_channels_to_maximize, ...
                                        riesz_transform_order, ...
                                        flag_debug_mode)
%RIESZANGLETEMPLATE steer Riesz coefficients for a given wavelet band
% in order to maximize the response of a template
%
% --------------------------------------------------------------------------
% Input arguments:
% 
% ORIG the original Riesz-wavelet coefficients at a given scale
%
% ORDER the order of the Riesz transform
%
% TEMPLATE coefficients for the linear combination of Riesz channels for which
% the response is to be maximized
%
% DEBUG Display some computation details if 1. Optional. Default is 0;
% orig structure of Riesz-wavelet coefficients
%
% --------------------------------------------------------------------------
% Output arguments:
%
% TH matrix of angles that are estimated pointwise in the wavelet band.
%
% MX Riesz-wavelet coefficient steered with respect to the angles in TH.
%
% --------------------------------------------------------------------------
%
% Part of the Generalized Riesz-wavelet toolbox
%
% Author: Dimitri Van De Ville. Ecole Polytechnique Federale de Lausanne.
%
% Version: Feb. 7, 2012

if nargin < 4,
    flag_debug_mode = 0;
end

%load polynomial coefficients:
matfn = sprintf('RieszAngle%d-Flex.dat',riesz_transform_order);
mat = reshape(load(matfn),[riesz_transform_order+1 riesz_transform_order+1 riesz_transform_order+1]);
mat = shiftdim(mat,1);

if flag_debug_mode,
    % visualize polynomial
    fprintf('\n');
    for degree_counter = 1:riesz_transform_order+1, % degree
        fprintf('tan(t)^%d ( ',riesz_transform_order+1-degree_counter);
        for channel_counter = 1:riesz_transform_order+1, % channel
            fprintf('+ ch[%d] ( ',channel_counter);
            for template_counter = 1:riesz_transform_order+1, % template
                if mat(template_counter,channel_counter,degree_counter),
                    fprintf('%+3.1f tm[%d] ',mat(template_counter,channel_counter,degree_counter),template_counter);
                end;
            end;
            fprintf(')\n    ');
        end;
        fprintf(')\n');
    end;
end;

%load steering matrix:
steermatfn = sprintf('RieszSteer%d.dat',riesz_transform_order);
steermat = shiftdim(reshape(load(steermatfn),[riesz_transform_order+1 riesz_transform_order+1 riesz_transform_order+1]),1);

terms = zeros(size(riesz_matrix_3D));
for degree_counter = 1:riesz_transform_order+1, % degree
    
    for channel_counter = 1:riesz_transform_order+1, % channel
        for template_counter = 1:riesz_transform_order+1, % template
            if mat(template_counter,channel_counter,degree_counter),
                terms(:,:, degree_counter) = terms(:,:,degree_counter) +...
                         riesz_matrix_3D(:,:,channel_counter) .* ...
                         linear_combination_coefficients_of_riesz_channels_to_maximize(template_counter) .* ...
                         mat(template_counter,channel_counter,degree_counter);
            end;
        end;
    end;
    
end


%compute th and mx:
steering_angles_matrix = zeros(size(terms, 1), size(terms, 2));
steered_riesz_matrix_3D = zeros(size(terms, 1), size(terms, 2));
for sample_counter1 = 1:size(terms, 1),
    parfor sample_counter2 = 1:size(terms, 2),
        C = zeros(1,riesz_transform_order+1); %C contains the increasing order terms at position (x1, x2)
        for degree_counter = 1:riesz_transform_order+1,
            C(degree_counter) = terms(sample_counter1, sample_counter2, degree_counter);
        end
        
        R = roots(C); %compute the roots of the polynomial C(1)*X^N + ... + C(N)*X + C(N+1)
        R = real(R(find(abs(imag(R))<1e-5))); %convert almost real roots to real numbers
        if isempty(R)
            R = 0;
        end;
        
        tha = atan(R); %compute arctangent of the roots
        costha = cos(tha);
        sintha = sin(tha);
        cossintha = cell(1,riesz_transform_order+1);
        for degree_counter2 = 1:riesz_transform_order+1, %build polynomial of costha and sintha
            cossintha{degree_counter2} = costha.^(riesz_transform_order-degree_counter2+1) .* ...
                                         sintha.^(degree_counter2-1);
        end;
        
        V = zeros(size(tha));
        for template_counter2 = 1:riesz_transform_order+1,           % template (row)
            for channel_counter2 = 1:riesz_transform_order+1,       % channel (column)
                for degree_counter2 = 1:riesz_transform_order+1, % term
                    V = V + ...
                        linear_combination_coefficients_of_riesz_channels_to_maximize(template_counter2) * ...
                        steermat(template_counter2,channel_counter2,degree_counter2) * ...
                        cossintha{degree_counter2} .* ...
                        riesz_matrix_3D(sample_counter1,sample_counter2, channel_counter2);
                end;
            end;
        end;
        
        idx = find(abs(V(:))==max(abs(V(:))));
        steering_angles_matrix(sample_counter1,sample_counter2) = -tha(idx(1));
        steered_riesz_matrix_3D(sample_counter1,sample_counter2) = max(abs(V(:)));
    end;
end;
