classdef(Enumeration) IsotropicWaveletType
    %ISOTROPICWAVELETTYPE enumeration of the types of
    % available functions for the radial part of the isotropic
    % and bandlimited wavelet transform
    %
    % --------------------------------------------------------------------------
    %
    % Part of the Generalized Riesz-wavelet toolbox
    %
    % Author: Nicolas Chenouard. Ecole Polytechnique Federale de Lausanne.
    %
    % Version: Feb. 7, 2012
    
    enumeration
        Simoncelli
        Shannon
        Aldroubi
        Papadakis
        Meyer
        Ward
    end
    methods (Static = true)
        function retVal = getDefaultValue()
            retVal = IsotropicWaveletType.Simoncelli;
        end
    end
end