function [QMF,dual_QMF] = wavelet_get_biorthonormal_QMF_filter_pair(type_string,p_parameter)
% MakeBSFilter -- Generate Biorthonormal QMF Filter Pairs
%  Usage
%    [qmf,dqmf] = MakeBSFilter(Type,Par)
%  Inputs
%    Type   string, one of:
%             'Triangle'
%             'Interpolating' 'Deslauriers' (the two are same)
%             'Average-Interpolating'
%             'CDF' (spline biorthogonal filters in Daubechies's book)
%             'Villasenor' (Villasenor's 5 best biorthogonal filters)
%    Par    integer list, e.g. if Type ='Deslauriers', Par=3 specifies
%           Deslauriers-Dubuc filter, polynomial degree 3
%  Outputs
%    qmf    quadrature mirror filter  (odd length, symmetric)
%    dqmf   dual quadrature mirror filter  (odd length, symmetric)
%
%  See Also
%    FWT_PBS, IWT_PBS, FWT2_PBS, IWT2_PBS
%
%  References
%    I. Daubechies, "Ten Lectures on Wavelets."
%
%    G. Deslauriers and S. Dubuc, "Symmetric Iterative Interpolating Processes."
%
%    D. Donoho, "Smooth Wavelet Decompositions with Blocky Coefficient Kernels."
%
%    J. Villasenor, B. Belzer and J. Liao, "Wavelet Filter Evaluation for
%    Image Compression."
%

if nargin < 2,
    p_parameter = 0;
end

sqr2 = sqrt(2);

if strcmp(type_string,'Triangle'),
    QMF = [0 1 0];
    dual_QMF = [.5 1 .5];

elseif strcmp(type_string,'Interpolating') | strcmp(type_string,'Deslauriers'),
    QMF  = [0 1 0];
    dual_QMF = wavelet_make_interpolating_refinement_filter(p_parameter)';
    dual_QMF =  dual_QMF(1:(length(dual_QMF)-1));

elseif strcmp(type_string,'Average-Interpolating'),
    QMF  = [0 .5 .5] ;
    dual_QMF = [0 ; wavelet_make_average_interpolating_filter(p_parameter)]';

elseif strcmp(type_string,'CDF'),
    if p_parameter(1)==1,
        dual_QMF = [0 .5 .5] .* sqr2;
        if p_parameter(2) == 1,
            QMF = [0 .5 .5] .* sqr2;
        elseif p_parameter(2) == 3,
            QMF = [0 -1 1 8 8 1 -1] .* sqr2 / 16;
        elseif p_parameter(2) == 5,
            QMF = [0 3 -3 -22 22 128 128 22 -22 -3 3].*sqr2/256;
        end
    elseif p_parameter(1)==2,
        dual_QMF = [.25 .5 .25] .* sqr2;
        if p_parameter(2)==2,
            QMF = [-.125 .25 .75 .25 -.125] .* sqr2;
        elseif p_parameter(2)==4,
            QMF = [3 -6 -16 38 90 38 -16 -6 3] .* (sqr2/128);
        elseif p_parameter(2)==6,
            QMF = [-5 10 34 -78 -123 324 700 324 -123 -78 34 10 -5 ] .* (sqr2/1024);
        elseif p_parameter(2)==8,
            QMF = [35 -70 -300 670 1228 -3126 -3796 10718 22050 ...
                10718 -3796 -3126 1228 670 -300 -70 35 ] .* (sqr2/32768);
        end
    elseif p_parameter(1)==3,
        dual_QMF = [0 .125 .375 .375 .125] .* sqr2;
        if p_parameter(2) == 1,
            QMF = [0 -.25 .75 .75 -.25] .* sqr2;
        elseif p_parameter(2) == 3,
            QMF = [0 3 -9 -7 45 45 -7 -9 3] .* sqr2/64;
        elseif p_parameter(2) == 5,
            QMF = [0 -5 15 19 -97 -26 350 350 -26 -97 19 15 -5] .* sqr2/512;
        elseif p_parameter(2) == 7,
            QMF = [0 35 -105 -195 865 363 -3489 -307 11025 11025 -307 -3489 363 865 -195 -105 35] .* sqr2/16384;
        elseif p_parameter(2) == 9,
            QMF = [0 -63 189 469 -1911 -1308 9188 1140 -29676 190 87318 87318 190 -29676 ...
                1140 9188 -1308 -1911 469 189 -63] .* sqr2/131072;
        end
    elseif p_parameter(1)==4,
        dual_QMF = [.026748757411 -.016864118443 -.078223266529 .266864118443 .602949018236 ...
            .266864118443 -.078223266529 -.016864118443 .026748757411] .*sqr2;
        if p_parameter(2) == 4,
            QMF = [0 -.045635881557 -.028771763114 .295635881557 .557543526229 ...
                .295635881557 -.028771763114 -.045635881557 0] .*sqr2;
        end
    end

elseif strcmp(type_string,'Villasenor'),
    if p_parameter == 1,
        % The "7-9 filters"
        QMF = [.037828455506995 -.023849465019380 -.11062440441842 .37740285561265];
        QMF = [QMF .85269867900940 reverse(QMF)];
        dual_QMF = [-.064538882628938 -.040689417609558 .41809227322221];
        dual_QMF = [dual_QMF .78848561640566 reverse(dual_QMF)];
    elseif p_parameter == 2,
        QMF  = [-.008473 .003759 .047282 -.033475 -.068867 .383269 .767245 .383269 -.068867...
            -.033475 .047282 .003759 -.008473];
        dual_QMF = [0.014182  0.006292 -0.108737 -0.069163 0.448109 .832848 .448109 -.069163 -.108737 .006292 .014182];
    elseif p_parameter == 3,
        QMF  = [0 -.129078 .047699 .788486 .788486 .047699 -.129078];
        dual_QMF = [0 .018914 .006989 -.067237 .133389 .615051 .615051 .133389 -.067237 .006989 .018914];
    elseif p_parameter == 4,
        QMF  = [-1 2 6 2 -1] / (4*sqr2);
        dual_QMF = [1 2 1] / (2*sqr2);
    elseif p_parameter == 5,
        QMF  = [0 1 1]/sqr2;
        dual_QMF = [0 -1 1 8 8 1 -1]/(8*sqr2);
    end
end