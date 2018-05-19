function [cropped_filters, filters_non_zero_indices] = get_filters_and_indices_which_are_non_zero( filters )
% [CROPPEDFILTERS, FILTIDX] = getFilterIDX(FILTERS)
% 
% FILTIDX{k} is the set of indices in FILTERS{k} that correspond to
% non-zero values and CROPPEDFILTERS{k} is FILTERS{k} cropped to this set
% of indices. See getIDXFromFilter.
% This allows for more efficient processing in building and reconstruction
% of the pyramid.
%
% Based on buildSCFpyr in matlabPyrTools
%
% Authors: Neal Wadhwa
% License: Please refer to the LICENCE file
% Date: July 2013
%

    number_of_filters = max(size(filters));
    filters_non_zero_indices = cell(number_of_filters, 2);    
    cropped_filters = cell(number_of_filters,1);
     
    for k = 1:number_of_filters
        indices = get_filters_row_and_column_indices_above_zero(filters{k});
        filters_non_zero_indices{k,1} = indices{1};
        filters_non_zero_indices{k,2} = indices{2};
        cropped_filters{k} = filters{k}(indices{1}, indices{2});
    end

end

