function res = reconstruct_steerable_pyramid_level_recursively(pyramid,index_matrix,lofilt,bfilts,boundary_conditions_string,levels,bands)
% RES = reconSpyrLevs(PYR,INDICES,LOFILT,BFILTS,EDGES,LEVS,BANDS)
%
% Recursive function for reconstructing levels of a steerable pyramid
% representation.  This is called by reconSpyr, and is not usually
% called directly.


number_of_bands = size(bfilts,2);
lo_ind = number_of_bands+1;
res_sz = index_matrix(1,:);

% Assume square filters:
bfiltsz =  round(sqrt(size(bfilts,1)));

if any(levels > 1)
    
    if  (size(index_matrix,1) > lo_ind)
        nres = reconstruct_steerable_pyramid_level_recursively( ...
                        pyramid(1+sum(prod(index_matrix(1:lo_ind-1,:)')):size(pyramid,1)),  ...
                        index_matrix(lo_ind:size(index_matrix,1),:), ...
                        lofilt, ...
                        bfilts, ...
                        boundary_conditions_string, ...
                        levels-1, ...
                        bands);
    else
        nres = get_pyramid_subband(pyramid,index_matrix,lo_ind); 	% lowpass subband
    end
    
    res = upsample_inserting_zeros_convolve(nres, lofilt, boundary_conditions_string, [2 2], [1 1], res_sz);
    
else
    
    res = zeros(res_sz);
    
end

if any(levels == 1)
    ind = 1;
    for b = 1:number_of_bands
        if any(bands == b)
            bfilt = reshape(bfilts(:,b), bfiltsz, bfiltsz);
            res = upsample_inserting_zeros_convolve(reshape(pyramid(ind:ind+prod(res_sz)-1), res_sz(1), res_sz(2)), ...
                bfilt, boundary_conditions_string, [1 1], [1 1], res_sz, res);
        end
        ind = ind + prod(res_sz);
    end
end
