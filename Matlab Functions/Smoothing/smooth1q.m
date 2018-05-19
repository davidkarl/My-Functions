function [z,s] = smooth1q(y,s,varargin)

%SMOOTH1Q Quick & easy smoothing.
%   Z = SMOOTH1Q(Y,S) smoothes the data Y using a DCT- or FFT-based spline
%   smoothing method. Non finite data (NaN or Inf) are treated as missing
%   values.
%
%   S is the smoothing parameter. It must be a real positive scalar. The
%   larger S is, the smoother the output will be. If S is empty (i.e. S =
%   []), it is automatically determined by minimizing the generalized
%   cross-validation (GCV) score.
%
%   Z = SMOOTH1Q(...,'robust') carries out a robust smoothing that
%   minimizes the influence of outlying data.
%
%   Z = SMOOTH1Q(...,'periodic') assumes that the data to be smoothed must
%   be periodic.
%
%   [Z,S] = SMOOTH1Q(...) also returns the calculated value for the
%   smoothness parameter S so that you can fine-tune the smoothing
%   subsequently if required.
%
%   SMOOTH1Q is a simplified and quick version of SMOOTHN for 1-D data. If
%   you want to smooth N-D arrays use <a
%   href="matlab:web('http://www.mathworks.com/matlabcentral/fileexchange/25634')">SMOOTHN</a>.
%
%   Notes
%   -----
%   1) SMOOTH1Q works with regularly spaced data only. Use SMOOTH1 for non
%      regularly spaced data.
%   2) The smoothness parameter used in this algorithm is determined
%      automatically by minimizing the generalized cross-validation score.
%      See the references for more details.
%
%   References
%   ----------
%   1) Garcia D, Robust smoothing of gridded data in one and higher
%   dimensions with missing values. Computational Statistics & Data
%   Analysis, 2010.
%   <a
%   href="matlab:web('http://www.biomecardio.com/pageshtm/publi/csda10.pdf')">PDF download</a>
%   2) Buckley MJ, Fast computation of a discretized thin-plate smoothing
%   spline for image data. Biometrika, 1994.
%   <a
%   href="matlab:web('http://biomet.oxfordjournals.org/content/81/2/247')">Link</a>
%
%   Examples:
%   --------
%   % Simple curve
%   x = linspace(0,100,200);
%   y = cos(x/10)+(x/50).^2 + randn(size(x))/10;
%   z = smooth1q(y,[]);
%   plot(x,y,'r.',x,z,'k','LineWidth',2)
%   axis tight square
%
%   % Periodic curve with ouliers and missing data
%   x = linspace(0,2*pi,300);
%   y = cos(x)+ sin(2*x+1).^2 + randn(size(x))/5;
%   y(150:155) = rand(1,6)*5;
%   y(10:40) = NaN;
%   subplot(121)
%   z = smooth1q(y,1e3,'periodic');
%   plot(x,y,'r.',x,z,'k','LineWidth',2)
%   axis tight square
%   title('Non robust')
%   subplot(122)
%   z = smooth1q(y,1e3,'periodic','robust');
%   plot(x,y,'r.',x,z,'k','LineWidth',2)
%   axis tight square
%   title('Robust')
%
%   % Limaçon
%   t = linspace(0,2*pi,300);
%   x = cos(t).*(.5+cos(t)) + randn(size(t))*0.05;
%   y = sin(t).*(.5+cos(t)) + randn(size(t))*0.05;
%   z = smooth1q(complex(x,y),[],'periodic');
%   plot(x,y,'r.',real(z),imag(z),'k','linewidth',2)
%   axis equal tight
%
%   See also SMOOTHN, SMOOTH1.
%
%   -- Damien Garcia -- 2012/08, revised 2014/02/26
%   website: <a
%   href="matlab:web('http://www.biomecardio.com')">www.BiomeCardio.com</a>

%-- Check input arguments
error(nargchk(2,4,nargin));
assert(isvector(squeeze(y)),...
    ['Y must be a 1-D array. Use <a href="matlab:web(''',...
    'http://www.mathworks.com/matlabcentral/fileexchange/25634'')">SMOOTHN</a> for non vector arrays.'])
if isempty(s)
    isauto = 1;
else
    assert(isnumeric(s),'S must be a numeric scalar')
    assert(isscalar(s) && s>0,...
        'The smoothing parameter S must be a scalar >0')
    isauto = 0;
end

%-- Order (use m>=2, m = 2 is recommended)
m = 2; % Note: order of the smoothing process, can be modified    

%-- Options ('robust' and/or 'periodic')
isrobust = 0; method = 'dct'; % default options
%--
if nargin>2
    assert(all(cellfun(@ischar,varargin)),...
        'The options must be ''robust'' and/or ''periodic''.')
    varargin = lower(varargin);
    if nargin==3
        idx = ismember({'robust','periodic'},varargin);
        assert(any(idx),...
            'The options must be ''robust'' and/or ''periodic''.')
        if idx(1), isrobust = 1; else method = 'fft'; end
    else % nargin = 4
        assert(all(ismember(varargin,{'robust','periodic'})),...
            'The options must be ''robust'' and/or ''periodic''.')
        isrobust = 1;
        method = 'fft';
    end
end

n = length(y);
siz0 = size(y);
y = y(:).';

%-- Weights
W0 = ones(siz0);
I = isfinite(y); % missing data (NaN or Inf values)
if any(~I) % replace the missing data (for faster convergence)
    X = 1:n;
    x = X(I); xi = X(~I);
    y(~I) = interp1(x,y(I),xi,'linear','extrap');
end
W0(~I) = 0; % weights for missing data are 0
W = W0;

%-- Eigenvalues
switch method
    case 'dct'
        Lambda = 2-2*cos((0:n-1)*pi/n);
    case 'fft'
        Lambda = 2-2*cos(2*(0:n-1)*pi/n);
end

%-- Smoothing process
nr = 3; % Number of robustness iterations
for k = 0:nr*isrobust
    if isrobust && k>0
        tmp = sqrt(1+16*s);
        h = sqrt(1+tmp)/sqrt(2)/tmp;
        W = W0.*bisquare(y,z,I,h);
    end
    if ~all(W==1) % then use an iterative method
        tol = Inf;
        zz = y;
        while tol>1e-3
            switch method
                case 'dct'
                    Y = dct(W.*(y-zz)+zz);
                case 'fft'
                    Y = fft(W.*(y-zz)+zz);
            end
            if isauto
                fminbnd(@GCVscore,-10,30,optimset('TolX',.1));
            else
                Gamma = 1./(1+s*Lambda.^m);
                switch method
                    case 'dct'
                        z = idct(Gamma.*Y);
                    case 'fft'
                        if isreal(y)
                            z = ifft(Gamma.*Y,'symmetric');
                        else
                            z = ifft(Gamma.*Y);
                        end
                end
            end
            tol = norm(zz-z)/norm(z);
            zz = z;
        end
        
    else %---
         % No missing values, non robust method => Direct fast method
         %---
        switch method
            case 'dct'
                Y = dct(y);
            case 'fft'
                Y = fft(y);
        end
        if isauto
            fminbnd(@GCVscore,-10,30,optimset('TolX',.1));
        else
            Gamma = 1./(1+s*Lambda.^m);
        end
        switch method
            case 'dct'
                z = idct(Gamma.*Y);
            case 'fft'
                if isreal(y)
                    z = ifft(Gamma.*Y,'symmetric');
                else
                    z = ifft(Gamma.*Y);
                end
        end
    end
end

z = reshape(z,siz0);

    function GCVs = GCVscore(p)
        s = 10^p;
        Gamma = 1./(1+s*Lambda.^m);
        if any(W)
            switch method
                case 'dct'
                    z = idct(Gamma.*Y);
                case 'fft'
                    if isreal(y)
                        z = ifft(Gamma.*Y,'symmetric');
                    else
                        z = ifft(Gamma.*Y);
                    end
            end
            RSS = norm(sqrt(W).*(y-z))^2;
        else % No missing values, non robust method => Direct fast method
            RSS = norm(Y.*(Gamma-1))^2;
        end
        TrH = sum(Gamma);
        GCVs = RSS/(1-TrH/n)^2;
    end

end

function W = bisquare(y,z,I,h)
r = y-z; % residuals
MAD = median(abs(r(I)-median(r(I)))); % median absolute deviation
u = abs(r/(1.4826*MAD)/sqrt(1-h)); % studentized residuals
W = (1-(u/4.685).^2).^2.*((u/4.685)<1); % bisquare weights
end