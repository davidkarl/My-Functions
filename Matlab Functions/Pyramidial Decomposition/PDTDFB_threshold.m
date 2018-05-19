function PDTDFB_in = PDTDFB_threshold(PDTDFB_in, threshold_method, threshold_parameter_vec, cfg)
% PDTDFB_THRSH : Threshold the PDTDFB data structure
%        yth = pdtdfb_thrsh(y, method, prm, [range])
% Keep a number of high coefficient in the directional subband of the
% PDTDFB data structure
%
% Input:
%   y       :  PDTDFB data structure
%   in      :  index value matrix
%   method  :  method of reset the coefficient
%       'threshold'     Keep coefficient bigger than prm
%   prm     paramater to complement method
%       'threshold'     prm is the threshold
%
% Output:
%   ang:	the angle of direction of the band in radian
%
% Note: BY default, the considered dfb level is always 2 in the pdtdfb data
% structure y
%
% See also: KEEPTOP, FINDTOP,  MKZERO_PDTDFB, PDTDFB_STAT

L = length(PDTDFB_in);
if length(threshold_parameter_vec) ~= L
    disp('Length prm should be the same as y');
end

if iscell(PDTDFB_in{end})
    %no residual band:
    for in1 = 2:L
        thresh = threshold_parameter_vec(in1);
        if iscell(PDTDFB_in{in1}{1})
            %pdtdfb data
            for in2 = 1:length(PDTDFB_in{in1}{1});
                tmp = abs(PDTDFB_in{in1}{1}{in2} + 1j*PDTDFB_in{in1}{2}{in2});
                idx = tmp < thresh;
                PDTDFB_in{in1}{1}{in2}(idx) = 0;
                PDTDFB_in{in1}{2}{in2}(idx) = 0;
            end
        else
            %wavelet data:
            for in2 = 1:length(PDTDFB_in{in1});
                tmp = abs(PDTDFB_in{in1}{in2});
                idx = tmp < thresh;
                PDTDFB_in{in1}{in2}(idx) = 0;
            end
        end
    end
    
else
    %residual band:
    for in1 = 2:L-1
        thresh = threshold_parameter_vec(in1);
        if iscell(PDTDFB_in{in1}{1})
            %pdtdfb data:
            for in2 = 1:length(PDTDFB_in{in1}{1});
                tmp = abs(PDTDFB_in{in1}{1}{in2} + 1j*PDTDFB_in{in1}{2}{in2});
                idx = tmp < thresh;
                PDTDFB_in{in1}{1}{in2}(idx) = 0;
                PDTDFB_in{in1}{2}{in2}(idx) = 0;
            end
        else
            %wavelet data:
            for in2 = 1:length(PDTDFB_in{in1});
                tmp = abs(PDTDFB_in{in1}{in2});
                idx = tmp < thresh;
                PDTDFB_in{in1}{in2}(idx) = 0;
            end
        end
    end
    tmp = abs(PDTDFB_in{L});
    thresh = threshold_parameter_vec(L);
    idx = tmp < thresh;
    PDTDFB_in{L}(idx) = 0;
end


