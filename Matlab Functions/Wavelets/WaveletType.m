classdef WaveletType
    %ISOTROPICWAVELETTYPE enumeration of the types of
    % available wavelet transforms
    %
    % --------------------------------------------------------------------------
    %
    % Part of the Generalized Riesz-wavelet toolbox
    %
    % Author: Nicolas Chenouard. Ecole Polytechnique Federale de Lausanne.
    %
    % Version: Feb. 7, 2012
    
    enumeration
        spline
        isotropic
    end
    methods (Static = true)
        function retVal = getDefaultValue()
            retVal = WaveletType.isotropic;
        end
    end
end

