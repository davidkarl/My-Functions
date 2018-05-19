function riesz_transform_object = riesz_transform_object_perpare_filters(riesz_transform_object)
% RIESZPREPARETRANSFORM2D Prepare the Riesz filters for Riesz pyramid in 2D
%
% config = RieszPrepareTransform2D(config) Prepare the Riesz filters for the
% Riesz pyramid in 2D corresponding to the RieszConfig2D object config.
% Returns a new RieszConfig object config containing the Riesz filters and
% the prefilter.
%
% --------------------------------------------------------------------------
% Input arguments:
%
% CONFIG RieszConfig2D object that specifies the Riesz-wavelet
% transform.
% 
% --------------------------------------------------------------------------
% Input arguments:
%
% CONFIG RieszConfig2D object that specifies the Riesz-wavelet
% transform. It contains filters for the Riesz tranform and for the
% prefiltering/postfiltering steps.
%
% --------------------------------------------------------------------------
%
% Part of the Generalized Riesz-wavelet toolbox
%
% Author: Dimitri Van De Ville and Nicolas Chenouard. Ecole Polytechnique Federale de Lausanne.
%
% Version: Feb. 7, 2012

[grid.yc, grid.xc] = ...
    ndgrid(2*pi*((1:riesz_transform_object.mat_size(1))-1)/riesz_transform_object.mat_size(1) - pi, ...
    2*pi*((1:riesz_transform_object.mat_size(2))-1)/riesz_transform_object.mat_size(2) - pi);


%% prepare prefilter:
flag_prefilter_ready = 0;
if (riesz_transform_object.prefilterType == PrefilterType.Shannon)
    [riesz_transform_object.prefilter.filterLow , riesz_transform_object.prefilter.filterHigh] = ...
                       get_shannon_prefilter(riesz_transform_object.mat_size(2), riesz_transform_object.mat_size(1), 1);
    flag_prefilter_ready = 1;
end
if (riesz_transform_object.prefilterType == PrefilterType.Simoncelli)
    [riesz_transform_object.prefilter.filterLow , riesz_transform_object.prefilter.filterHigh] = ...
                       get_raised_cosine_prefilter(riesz_transform_object.mat_size(2), riesz_transform_object.mat_size(1), 1);
    flag_prefilter_ready = 1;
end
if (riesz_transform_object.prefilterType == PrefilterType.None)
    riesz_transform_object.prefilter.filterLow = [];
    riesz_transform_object.prefilter.filterHigh = [];
    flag_prefilter_ready = 1;
end
if ~flag_prefilter_ready,
    error('wrong prefilter type. See PrefilterType enumeration.');
end

%% prepare spline wavelet decomposition:
if strcmpi(riesz_transform_object.wavelet_type,'spline'),
    error('2D spline wavelets are not included in this toolbox, please consider using isotropic wavelets instead')
    %config = prepareSplineWavelets2(config, grid);
end;

%prepare Riesz transform:
if riesz_transform_object.riesz_transform_order>0, % enable Riesz transform
     
    tmp = 1./sqrt(grid.xc.^2+grid.yc.^2);
    
    %number of components:
    number_of_dimensions = 2;
    riesz_transform_object.number_of_riesz_channels = nchoosek(number_of_dimensions+riesz_transform_object.riesz_transform_order-1 , ...
                                                                        riesz_transform_object.riesz_transform_order);
    
    %1d filter: -j freq / norm of pulsation vector, and swap quadrants:
    base1 = ifftshift(-1i*grid.yc.*tmp);
    %set the 0 frequency to 1:
    base1(1,1) = 1;
    clear grid.yc;
    
    base2 = ifftshift(-1i*grid.xc.*tmp);
    %set the 0 frequency to 1:
    base2(1,1) = 1;
    clear grid.xc;
    
    %keep only the imaginary part of the central frequency as to alleviate aliasing:
    %UNDERSTAND THIS AGAIN, KNEW THIS ONCE!!!!
    base1(end/2+1,:) = imag(base1(end/2+1,:));
    base2(:,end/2+1) = imag(base2(:,end/2+1));
    
    riesz_transform_object.riesz_transform_filters = cell(1, riesz_transform_object.number_of_riesz_channels);
    %config.RieszOrders = cell(1, config.RieszChannels);
    
    iteration_counter = 1;
    for n1 = 0:riesz_transform_object.riesz_transform_order
        n2 = riesz_transform_object.riesz_transform_order - n1;
        riesz_transform_object.riesz_transform_filters{iteration_counter}  = ...
            sqrt(multinomial(riesz_transform_object.riesz_transform_order, n1, n2)) * base1.^(n1).*base2.^(n2);
        iteration_counter = iteration_counter+1;
    end;
    
    %normalize 0 frequency (INTERESTING TO UNDERSTAND!!!!):
    dN = number_of_dimensions^(0.5*riesz_transform_object.riesz_transform_order);
    for k = 1:riesz_transform_object.number_of_riesz_channels,
        riesz_transform_object.riesz_transform_filters{k}(1) = ...
            riesz_transform_object.riesz_transform_filters{k}(1)/dN;
    end
    
else %if riesz_transform_object.riesz_transform_order>0
    riesz_transform_object.number_of_riesz_channels = 1;
end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Auxiliary functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function c = multinomial(n,varargin)
% MULTINOMIAL Multinomial coefficients
%
%   MULTINOMIAL(N, K1, K2, ..., Km) where N and Ki are floating point
%   arrays of non-negative integers satisfying N = K1 + K2 + ... + Km,
%   returns the multinomial  coefficient   N!/( K1!* K2! ... *Km!).
%
%   MULTINOMIAL(N, [K1 K2 ... Km]) when Ki's are all scalar, is the
%   same as MULTINOMIAL(N, K1, K2, ..., Km) and runs faster.
%
%   Non-integer input arguments are pre-rounded by FLOOR function.
%
% EXAMPLES:
%    multinomial(8, 2, 6) returns  28
%    binomial(8, 2) returns  28
%
%    multinomial(8, 2, 3, 3)  returns  560
%    multinomial(8, [2, 3, 3])  returns  560
%
%    multinomial([8 10], 2, [6 8]) returns  [28  45]

% Mukhtar Ullah
% November 1, 2004
% mukhtar.ullah@informatik.uni-rostock.de

nIn = nargin;
error(nargchk(2, nIn, nIn))

if ~isreal(n) || ~isfloat(n) || any(n(:)<0)
    error('Inputs must be floating point arrays of non-negative reals')
end

arg2 = varargin;
dim = 2;

if nIn < 3
    k = arg2{1}(:).';
    if isscalar(k)
        error('In case of two arguments, the 2nd cannot be scalar')
    end
else
    [arg2{:},sizk] = sclrexpnd(arg2{:});
    if sizk == 1
        k = [arg2{:}];
    else
        if ~isscalar(n) && ~isequal(sizk,size(n))
            error('Non-scalar arguments must have the same size')
        end
        dim = numel(sizk) + 1;
        k = cat(dim,arg2{:});
    end
end

if ~isreal(k) || ~isfloat(k) || any(k(:)<0)
    error('Inputs must be floating point arrays of non-negative reals')
end

n = floor(n);
k = floor(k);

if any(sum(k,dim)~=n)
    error('Inputs must satisfy N = K1 + K2 ... + Km ')
end

c = floor(exp(gammaln(n+1) - sum(gammaln(k+1),dim)) + .5);
end




function [varargout] = sclrexpnd(varargin)
% SCLREXPND expands scalars to the size of non-scalars.
%    [X1,X2,...,Xn] = SCLREXPND(X1,X2,...,Xn) expands the scalar
%    arguments, if any, to the (common) size of the non-scalar arguments,
%    if any.
%
%    [X1,X2,...,Xn,SIZ] = SCLREXPND(X1,X2,...,Xn) also returns the
%    resulting common size in SIZ.
%
% Example:
% >> c1 = 1; c2 = rand(2,3); c3 = 5; c4 = rand(2,3);
% >> [c1,c2,c3,c4,sz] = sclrexpnd(c1,c2,c3,c4)
%
% c1 =
%      1     1     1
%      1     1     1
%
% c2 =
%     0.7036    0.1146    0.3654
%     0.4850    0.6649    0.1400
%
% c3 =
%      5     5     5
%      5     5     5
%
% c4 =
%     0.5668    0.6739    0.9616
%     0.8230    0.9994    0.0589
%
% sz =
%      2     3

% Mukhtar Ullah
% November 2, 2004
% mukhtar.ullah@informatik.uni-rostock.de

C = varargin;
issC = cellfun('prodofsize',C) == 1;
if issC
    sz = [1 1];
else
    nsC = C(~issC);
    if ~isscalar(nsC)
        for i = 1:numel(nsC), nsC{i}(:) = 0;  end
        if ~isequal(nsC{:})
            error('non-scalar arguments must have the same size')
        end
    end
    s = find(issC);
    sz = size(nsC{1});
    for i = 1:numel(s)
        C{s(i)} = C{s(i)}(ones(sz));
    end
end
varargout = [C {sz}];
end