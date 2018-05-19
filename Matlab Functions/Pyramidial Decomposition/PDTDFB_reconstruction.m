function x = PDTDFB_reconstruction(y, lpfname, dfname, wfname, res)
% PDTDFBREC   Pyramid Dual Tree Directional Filterbank Reconstruction
%
%	x = pdtdfbrec(y, res, [lpfname], [dfname], [wfname])
%
% Input:
%   y:	      a cell vector of length n+1, one for each layer of
%       	  subband images from DFB, y{1} is the low band image
%   lpfname:  [optional] The name of the laplacian lowpass filter that is used, default is 'meyer'
%             if lpfname is 'nalias' then the lpnadec and rec is used
%   dfname:   [optional] The name of the diamond or fan filter that is used,
%             see dilfters
%             'meyer': default (very big filters - very slow)
%             'pkva''pkva6', 'pkva8', 'pkva12' (Contourlet toolbox, using ladder structure)
%   wfname:   [optional] The name of the wavelet filter that is used, default is '9-7'
%   res:      [optional] Residual high passband, default 0 is no, 1 is yes
%   pftype:   Type of the pyramidal filter decomposition (LPDEC parameter)
%
%
% Output:
%   x:      reconstructed image
%
% Call function: LPREC,
% See also:	PDTDFBREC, PDFBDEC, TDFBDEC,
%
% Note : PDTDFB data structure y{resolution}{1}{1-2^n} : primal branch
%                              y{resolution}{2}{1-2^n} : dual branch

if ~exist('lpfname','var')
    lpfname = 'nalias' ; % default implementation by meyer type FB
end

if ~exist('dfname','var')
    dfname = 'meyer' ; % default implementation by meyer type FB
    % Note: meyer diamond is different from meyer type low pass
end

if ~exist('wfname','var')
    wfname = '9-7' ; % default implementation by meyer type FB
end

if ~exist('res','var')
    res = 0 ; % default implementation by meyer type FB
end

% determine the kind of laplacian pyramid decomposition based on
% lpfname decide type of pyramidal decomposition
switch lower(lpfname)
    case ('special')
        lptype = 3; %
        disp('Frequency implemenation for the first two level')
    case ('nalias')
        lptype = 2; % noaliasing type pyramid
        disp('Noalias frame pyramid')
    case ('fp')
        lptype = 0; % Burt and Aldeson pyramid
        disp('Frameming pyramid')
    otherwise
        lptype = 1; % Burt and Aldeson pyramid
        disp('Burt and Aldeson pyramid')
end

n = length(y) - 1;
if n <= 0
    x = y{1};
    % exit fucntion
else
    % no residual band on recursive call
    res2 = 0;
    
    % Recursive call to reconstruct the low band
    xlo = PDTDFB_reconstruction(y(1:end-1), lpfname, dfname, wfname, res2);
    
    if res % residual band processing
        % processing residual band at the last step,
        x_res = y{end};
        
        % estimating filter
        N = 128;
        cutoff = 2*pi;
        % cutoff =    pi;
        rev = (8*pi^2/3)/cutoff;
        xa = 0:rev/(N):rev;
        % Compute support of Fourier transform of phi.
        int1 = find((xa < 2*pi/3));
        int2 = find((xa >= 2*pi/3) & (xa < 4*pi/3));
        
        % Compute Fourier transform of phi.
        phihat = zeros(1,N+1);
        phihat(int1) = ones(size(int1));
        phihat(int2) = cos(pi/2*meyeraux(3/2/pi*xa(int2)-1));
        
        phihat = [phihat,phihat((end-1):-1:2)];
        psihat = (1-phihat).^(1/2);
        h = ifftshift(ifft(phihat));
        g = ifftshift(ifft(psihat));
        
        % [h,g] = wfilters('dmey','l');
        h = fitmat(h,[1,32]); h = h(2:end);
        h = h./sum(h);
        H = h'*h;
        G = fftshift(ifft2( (1 - abs(fft2(H)).^2).^(1/2) ) );
        G = fitmat(real(G),31);
        
        % filtering and add
        xlo = filter_2D_with_edge_handling(xlo,H,'sym');
        x_res = filter_2D_with_edge_handling(x_res, G,'sym');
        x = xlo + x_res;
    else
        
        % Process the detail subbands
        % ---------------------------------------------------------------
        if ~iscell(y{end}) % if not cell , then it must be higpass laplacian (not decomposed by dfb)
            % laplacian pyramid
            % Get the pyramidal filters
            [h,g] = get_filters_for_laplacian_pyramid(lpfname);
            
            x = laplacian_pyramid_decomposition(xlo, y{end}, h, g, pftype);
            % ---------------------------------------------------------------
        elseif (length(y{end}) ~= 3 ) && (length(y{end}) ~= 1) % must be dual tree dfb
            if (lptype~=3)
                % Decide the method based on the filter name
                switch dfname
                    case {'pkva6', 'pkva8', 'pkva12', 'pkva'}
                        % Use the ladder structure (much more efficient)
                        pr = DFB_reconstruction_time_domain_ladder_structure(y{end}{1},'primal', dfname);
                        dl = DFB_reconstruction_time_domain_ladder_structure(y{end}{2},'dual', dfname);
                        % disp('ladder');
                        xhi = 0.5*(pr + dl);
                        % xhi = pr;
                    otherwise
                        % General case
                        % Reconstruct the bandpass image from DFB
                        pr = DFB_reconstruction_time_domain(y{end}{1},'primal', dfname);
                        dl = DFB_reconstruction_time_domain(y{end}{2},'dual', dfname);
                        xhi = 0.5*(pr + dl);
                end
                % Get the pyramidal filters
                [h,g] = get_filters_for_laplacian_pyramid(lpfname);
                x = laplacian_pyramid_reconstruction(xlo, xhi, h, g, lptype);
                
            else
                
                % frequency implementation
                % window estimation, the same for the forward transform
                % except size estimation
                alpha = 0.15;
                % estimate based on conventional ordering of subband
                s = 2*[size(y{end}{1}{end}, 1),  size(y{end}{1}{1}, 2)];
                
                % create the grid and transform to circle grid
                S1 = -1.5*pi:pi/(s(1)/2):1.5*pi;
                S2 = -1.5*pi:pi/(s(2)/2):1.5*pi;
                [x1, x2] = meshgrid(S2,S1);
                r = [0.4 0.5 1-alpha, 1+ alpha];
                
                [x1, x2]=tran_sf(x1,x2);
                
                rd = sqrt(x1.*x1+x2.*x2);
                theta = angle(x1+sqrt(-1)*x2);
                
                % Low pass window
                sz = size(rd);
                cen = rd((sz(1)+1)/2,:);
                cen = abs(cen);
                fl = get_1D_meyer_function(cen,pi*[-2 -1 r(1:2)]);
                FL =  fl'*fl;
                
                % high pass window
                ang = pi/4*[-alpha alpha];
                ang = [-pi/4+ang(1:2), 3*pi/4+ang(1:2)];
                
                f3 = fun_meyer_curv(rd, theta, pi*r, ang, 's');
                f3 = periodize_2D(f3,s, 's');
                
                % take out the center and square root
                FL = sqrt(fftshift(FL(s(1)/4+1:s(1)/4+s(1),s(2)/4+1:s(2)/4+s(2))));
                f3 = sqrt(fftshift(f3(s(1)/4+1:s(1)/4+s(1),s(2)/4+1:s(2)/4+s(2))));
                
                % actual transform by FFT, different from dec
                
                % Use the ladder structure (much more efficient)
                pr = DFB_reconstruction_time_domain_ladder_structure(y{end}{1},'primal', dfname);
                dl = DFB_reconstruction_time_domain_ladder_structure(y{end}{2},'primal', dfname);
                im2f = fft2(pr+sqrt(-1)*dl);
                
                % handle low resolution image
                sz = s/2;
                dm = [2 2];
                
                im1f = kron(ones(dm),fft2(xlo));
                
                x = 2*real(ifft2(0.5*prod(dm)*im1f.*FL + im2f.*f3) );
                
                
            end
            
            % ---------------------------------------------------------------
        elseif (length(y{end}) == 3) % must be wavelet decomposition
            % Special case: length(y{end}) == 3
            % Perform one-level 2-D critically sampled wavelet filter bank
            [h, g] = get_filters_for_laplacian_pyramid(wfname);
            x = wavelet_filterbank_2D_reconstruction(xlo, y{end}{1}, y{end}{2}, y{end}{3}, h, g);
        else
            error('What ?');
        end % if
        
    end
    
end

