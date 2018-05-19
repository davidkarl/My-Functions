function resdft = reconstruct_steerable_pyramid_fourier_level_recursively(...
                        pyramid,index_matrix,log_rad,Xrcos,Yrcos,angle,number_of_bands,levels,bands)
% RESDFT = reconSFpyrLevs(PYR,INDICES,LOGRAD,XRCOS,YRCOS,ANGLE,NBANDS,LEVS,BANDS)
%
% Recursive function for reconstructing levels of a steerable pyramid
% representation.  This is called by reconSFpyr, and is not usually
% called directly.

lo_ind = number_of_bands+1;
current_pyramid_dimensions = index_matrix(1,:);
pyramid_center = ceil((current_pyramid_dimensions+0.5)/2);

%log_rad = log_rad + 1;
Xrcos = Xrcos - log2(2);  % shift origin of lut by 1 octave.

if any(levels > 1)
    
    lodims = ceil((current_pyramid_dimensions-0.5)/2);
    loctr = ceil((lodims+0.5)/2);
    lostart = pyramid_center-loctr+1;
    loend = lostart+lodims-1;
    nlog_rad = log_rad(lostart(1):loend(1),lostart(2):loend(2));
    nangle = angle(lostart(1):loend(1),lostart(2):loend(2));
    
    if size(index_matrix,1) > lo_ind
        nresdft = reconstruct_steerable_pyramid_fourier_level_recursively( ...
                        pyramid(1+sum(prod(index_matrix(1:lo_ind-1,:)')):size(pyramid,1)),...
                        index_matrix(lo_ind:size(index_matrix,1),:), ...
                        nlog_rad, Xrcos, Yrcos, nangle, number_of_bands,levels-1, bands);
    else
        nresdft = fftshift(fft2(get_pyramid_subband(pyramid,index_matrix,lo_ind)));
    end
    
    YIrcos = sqrt(abs(1.0 - Yrcos.^2));
    lomask = apply_point_operation_to_image(nlog_rad, YIrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);
    
    resdft = zeros(current_pyramid_dimensions);
    resdft(lostart(1):loend(1),lostart(2):loend(2)) = nresdft .* lomask;
    
else
    
    resdft = zeros(current_pyramid_dimensions);
    
end


if any(levels == 1)
    
    lutsize = 1024;
    Xcosn = pi*[-(2*lutsize+1):(lutsize+1)]/lutsize;  % [-2*pi:pi]
    order = number_of_bands-1;
    % divide by sqrt(sum_(n=0)^(N-1)  cos(pi*n/N)^(2(N-1)) )
    const = (2^(2*order))*(factorial(order)^2)/(number_of_bands*factorial(2*order));
    Ycosn = sqrt(const) * (cos(Xcosn)).^order;
    himask = apply_point_operation_to_image(log_rad, Yrcos, Xrcos(1), Xrcos(2)-Xrcos(1),0);
    
    ind = 1;
    for b = 1:number_of_bands
        if any(bands == b)
            anglemask = apply_point_operation_to_image(angle,Ycosn,Xcosn(1)+pi*(b-1)/number_of_bands,Xcosn(2)-Xcosn(1));
            band = reshape(pyramid(ind:ind+prod(current_pyramid_dimensions)-1), current_pyramid_dimensions(1), current_pyramid_dimensions(2));
            banddft = fftshift(fft2(band));
            resdft = resdft + (sqrt(-1))^(number_of_bands-1) * banddft.*anglemask.*himask;
        end
        ind = ind + prod(current_pyramid_dimensions);
    end
end

