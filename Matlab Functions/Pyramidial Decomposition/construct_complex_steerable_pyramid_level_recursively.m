function [pyr,pind] = construct_complex_steerable_pyramid_level_recursively(...
                                            lodft,log_rad,Xrcos,Yrcos,angle,height,number_of_bands)
% [PYR, INDICES] = buildSCFpyrLevs(LODFT, LOGRAD, XRCOS, YRCOS, ANGLE, HEIGHT, NBANDS)
%
% Recursive function for constructing levels of a steerable pyramid.  This
% is called by buildSCFpyr, and is not usually called directly.

% Modified by Javier Portilla to generate complex bands in 9/97.

if (height <= 0)  
    
    lo0 = ifft2(ifftshift(lodft));
    pyr = real(lo0(:));
    pind = size(lo0);
    
else
    
    bands = zeros(numel(lodft), number_of_bands);
    bind = zeros(number_of_bands,2);
    
    %  log_rad = log_rad + 1;
    Xrcos = Xrcos - log2(2);  % shift origin of lut by 1 octave.
    
    look_up_table_size = 1024;
    Xcosn = pi*[-(2*look_up_table_size+1):(look_up_table_size+1)]/look_up_table_size;  % [-2*pi:pi]
    order = number_of_bands-1;
    % divide by sqrt(sum_(n=0)^(N-1)  cos(pi*n/N)^(2(N-1)) )
    % Thanks to Patrick Teo for writing this out :)
    const = (2^(2*order))*(factorial(order)^2)/(number_of_bands*factorial(2*order));
    
    %
    %  Ycosn = sqrt(const) * (cos(Xcosn)).^order;
    %
    % analityc version: only take one lobe
    alfa = mod(pi+Xcosn,2*pi)-pi;
    Ycosn = 2*sqrt(const) * (cos(Xcosn).^order) .* (abs(alfa)<pi/2);
    
    himask = apply_point_operation_to_image(log_rad, Yrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);
    
    for band_counter = 1:number_of_bands
        anglemask = apply_point_operation_to_image(angle, Ycosn, Xcosn(1)+pi*(band_counter-1)/number_of_bands, Xcosn(2)-Xcosn(1));
        banddft = ((-1i)^(number_of_bands-1)) .* lodft .* anglemask .* himask;
        band = ifft2(ifftshift(banddft));
        
        %    bands(:,b) = real(band(:));
        % analytic version: full complex value
        bands(:,band_counter) = band(:);
        bind(band_counter,:) = size(band);
    end
    
    dims = size(lodft);
    ctr = ceil((dims+0.5)/2);
    lodims = ceil((dims-0.5)/2);
    loctr = ceil((lodims+0.5)/2);
    lostart = ctr-loctr+1;
    loend = lostart+lodims-1;
    
    log_rad = log_rad(lostart(1):loend(1),lostart(2):loend(2));
    angle = angle(lostart(1):loend(1),lostart(2):loend(2));
    lodft = lodft(lostart(1):loend(1),lostart(2):loend(2));
    YIrcos = abs(sqrt(1.0 - Yrcos.^2));
    lomask = apply_point_operation_to_image(log_rad, YIrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);
    
    lodft = lomask .* lodft;
    
    [npyr,nind] = construct_complex_steerable_pyramid_level_recursively(lodft, log_rad, Xrcos, Yrcos, angle, height-1, number_of_bands);
    
    pyr = [bands(:); npyr];
    pind = [bind; nind];
    
end

