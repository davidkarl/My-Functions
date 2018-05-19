function [channel_weights_vec,channel_weights_over_time,error_over_time] = ...
                                kalman_filter_MCAF(input_signal_matrix,desired_signal,varargin)
%
%[xhat,XHAT,ALPHA]=kallman(x,d,xhat0,lamb,K0,sig)

%[xhat,XHAT,ALPHA]=kallman(U,d,xhat0,lamb,K0,sig)
%
%Implements unforced kallman filter. Tap weights are the state vectors
%inputs are C(t). Parameters are:
% 
% U       - (NxM) Signals that will be used to predict d.
% d       - (Nx1) Desired response
% xhat0   - (Mx1) Initial Weights (optional)
% lamb    - (1x1) Forgetting Factor (optional)
% K0      - (Mx1) Initialization matrix (optional)
% sig     - (NxM) Noise estimates of each U channel (optional)
% 
% 
% xhat    - (1xM) Final estimated weights
% XHAT    - (NxM) Weight learning curve
% ALPHA   - (Nx1) Error curve



%Implements unforced kallman filter. Tap weights are the state vectors
%inputs are C(t)
% Copyright (C) 2010 Ikaro Silva
% 
% This library is free software; you can redistribute it and/or modify it under
% the terms of the GNU Library General Public License as published by the Free
% Software Foundation; either version 2 of the License, or (at your option) any
% later version.
% 
% This library is distributed in the hope that it will be useful, but WITHOUT ANY
% WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
% PARTICULAR PURPOSE.  See the GNU Library General Public License for more
% details.
% 
% You should have received a copy of the GNU Library General Public License along
% with this library; if not, write to the Free Software Foundation, Inc., 59
% Temple Place - Suite 330, Boston, MA 02111-1307, USA.
% 
% You may contact the author by e-mail (ikaro@ieee.org).

%Initialize all parameters
[samples_per_channel,number_of_channels] = size(input_signal_matrix);
lambda_smoothing_factor = 1;

if (nargin>2 && ~isempty(varargin{1}) )
    channel_weights_vec=varargin{1};
    channel_weights_vec=channel_weights_vec(:);
else
    channel_weights_vec=rand(number_of_channels,1)*0.0001;
end
if (nargin>3 && ~isempty(varargin{2}))
    lambda_smoothing_factor=varargin{2};
end
if (nargin>4 && ~isempty(varargin{3}))
    P_state_estimation_covariance=varargin{3};
else
    P_state_estimation_covariance = corrmtx(channel_weights_vec,number_of_channels-1);
    P_state_estimation_covariance = P_state_estimation_covariance'*P_state_estimation_covariance;
end
if (nargin>5 && ~isempty(varargin{4}))
    measurement_noise_matrix = varargin{4};
else
    measurement_noise_matrix = ones(samples_per_channel,number_of_channels);  %measureement noise (assumed diagonal)!!!!
end


%Do the training
lambda_smoothing_factor = lambda_smoothing_factor^(-0.5);
error_over_time = zeros(samples_per_channel,1);
channel_weights_over_time = zeros(samples_per_channel,number_of_channels);

for n = 1:samples_per_channel
    
    current_channels_sample = input_signal_matrix(n,:)';   
    
    den = diag(1./(current_channels_sample'*P_state_estimation_covariance*current_channels_sample + measurement_noise_matrix(n,:)));
    kalman_gain = lambda_smoothing_factor*den*P_state_estimation_covariance*current_channels_sample ;
    
    current_channel_weighting_error = desired_signal(n) - current_channels_sample'*channel_weights_vec;
    
    channel_weights_vec = lambda_smoothing_factor*channel_weights_vec + ...
                                                        kalman_gain*current_channel_weighting_error;
    
    P_state_estimation_covariance = (lambda_smoothing_factor^2)*P_state_estimation_covariance - ...
                lambda_smoothing_factor*kalman_gain*current_channels_sample'*P_state_estimation_covariance;

    error_over_time(n) = current_channel_weighting_error;
    channel_weights_over_time(n,:) = channel_weights_vec';
    
end





