function [median_vec,min_vec,max_vec,median_pointer,spectrum_buffer,Next_pointer_chain,Previous_pointer_chain]=running_median_calculation(current_spectrum,spectrum_buffer,Next_pointer_chain,Previous_pointer_chain,current_buffer_index,median_pointer,median_vec)
% 
% buffer_size=uint8(41); % must be odd;
% total_median_memory_size=42;
% Px=repmat([uint8(buffer_size+1),uint8(1):buffer_size],half_spectrum,1);
% Nx=repmat([uint8(2):buffer_size,buffer_size+1,uint8(1)],half_spectrum,1);
% med_ptr_x=ones(half_spectrum,1,'uint8')*uint8((buffer_size+1)/2);
% maximum_frequency_index_to_check=513;
% median_vec=zeros(maximum_frequency_index_to_check,1);

% median_vec = zeros(size(spectrum_buffer,1),1);
buffer_size = size(spectrum_buffer,2);
total_median_memory_size = buffer_size;
max_vec = zeros(size(spectrum_buffer,1),1);
min_vec = zeros(size(spectrum_buffer,1),1);
for frequency_index=1:size(current_spectrum,1)
    %get min and max indices:
    min_index = Next_pointer_chain(frequency_index,total_median_memory_size); %index of min
    max_index = Previous_pointer_chain(frequency_index,total_median_memory_size); %index of max
    
    %get previous spectrum values, and previous min and max values:
    previous_spectrum_value = spectrum_buffer(frequency_index,current_buffer_index);
    previous_minimum_spectrum_value = spectrum_buffer(frequency_index,min_index);
    previous_maximum_spectrum_value = spectrum_buffer(frequency_index,max_index);
    
    %get current (new) spectrum value:
    current_spectrum_value = current_spectrum(frequency_index);
    
    %inject current spectrum into spectrum buffer history:
    spectrum_buffer(frequency_index,current_buffer_index)=current_spectrum_value;
    
    %get old spectrum median value:
    old_median_value = spectrum_buffer(frequency_index,median_pointer(frequency_index));
    
    %if new value is a new minimum but the minimum index is already the current buffer index:
    %--> the chain is linked and the median is unchanged so go on to check next frequency:
    if current_spectrum_value<=previous_spectrum_value && min_index==current_buffer_index
        median_vec(frequency_index)=old_median_value;
        continue;
    end
    
    %if new value is a new maximum but the maximum index is already the current buffer index:
    %--> the chain is linked and the median is unchanged so go on to check next frequency:
    if current_spectrum_value>=previous_spectrum_value && max_index==current_buffer_index
        median_vec(frequency_index)=old_median_value;
        continue;
    end
    
    %the new value is the new min but not replacing previus min
    if (current_spectrum_value<=previous_minimum_spectrum_value && current_buffer_index~=min_index)
        %remove old index from the list
        Previous_pointer_chain(frequency_index,Next_pointer_chain(frequency_index,current_buffer_index))=Previous_pointer_chain(frequency_index,current_buffer_index);
        Next_pointer_chain(frequency_index,Previous_pointer_chain(frequency_index,current_buffer_index))=Next_pointer_chain(frequency_index,current_buffer_index);
        % if Oldval was smaller than median the median is unchanged but if
        % it is bigger than the old median the new median must shift to the
        % previous (smaller) value
        if previous_spectrum_value>=old_median_value
            median_pointer(frequency_index)=Previous_pointer_chain(frequency_index,median_pointer(frequency_index));
        end
        
        Next_pointer_chain(frequency_index,current_buffer_index)=min_index; % next bigger is the previous min
        Previous_pointer_chain(frequency_index,min_index)=current_buffer_index; % previous of the old minimum is the current ind
        Next_pointer_chain(frequency_index,total_median_memory_size)=current_buffer_index; % update min_ind
        Previous_pointer_chain(frequency_index,current_buffer_index)=total_median_memory_size; % mark this as the min
        
        % the chain is linked
        
        median_vec(frequency_index)=spectrum_buffer(frequency_index,median_pointer(frequency_index));
        continue;
    end
    
    %the new value is the new max but not replacing previus max
    if (current_spectrum_value>=previous_maximum_spectrum_value && current_buffer_index~=max_index) %
        %remove old ind from the list
        Previous_pointer_chain(frequency_index,Next_pointer_chain(frequency_index,current_buffer_index))=Previous_pointer_chain(frequency_index,current_buffer_index);
        Next_pointer_chain(frequency_index,Previous_pointer_chain(frequency_index,current_buffer_index))=Next_pointer_chain(frequency_index,current_buffer_index);
        % if Oldval was bigger than median the median is unchanged but if
        % it was smaller than the old median the new median must shift to the
        % next (bigger) value
        if previous_spectrum_value<=old_median_value
            median_pointer(frequency_index)=Next_pointer_chain(frequency_index,median_pointer(frequency_index));
        end
        
        Previous_pointer_chain(frequency_index,current_buffer_index)=max_index; % next smaller is the previous max
        Next_pointer_chain(frequency_index,max_index)=current_buffer_index; % next of the old max is the current ind
        Previous_pointer_chain(frequency_index,total_median_memory_size)=current_buffer_index; % update max_ind
        Next_pointer_chain(frequency_index,current_buffer_index)=total_median_memory_size; % mark this as the max
        
        % the chain is linked
        
        median_vec(frequency_index)=spectrum_buffer(frequency_index,median_pointer(frequency_index));
        continue;
    end
    
    %% if replacing the previeus medien
    if current_buffer_index==median_pointer(frequency_index)
        % then if the new value is  bigger than the next value after the old median
        % than the new median is the next of the old median
        if current_spectrum_value>=spectrum_buffer(frequency_index,Next_pointer_chain(frequency_index,median_pointer(frequency_index)))
            %% remove old ind from the list
            Previous_pointer_chain(frequency_index,Next_pointer_chain(frequency_index,current_buffer_index))=Previous_pointer_chain(frequency_index,current_buffer_index);
            Next_pointer_chain(frequency_index,Previous_pointer_chain(frequency_index,current_buffer_index))=Next_pointer_chain(frequency_index,current_buffer_index);
            median_pointer(frequency_index)=Next_pointer_chain(frequency_index,median_pointer(frequency_index));
            [Next_pointer_chain,Previous_pointer_chain]=search_and_replace_up(median_pointer(frequency_index),current_buffer_index,Next_pointer_chain,Previous_pointer_chain,spectrum_buffer,current_spectrum_value,total_median_memory_size,frequency_index);
        elseif current_spectrum_value<=spectrum_buffer(frequency_index,Previous_pointer_chain(frequency_index,median_pointer(frequency_index)))
            %% remove old ind from the list
            Previous_pointer_chain(frequency_index,Next_pointer_chain(frequency_index,current_buffer_index))=Previous_pointer_chain(frequency_index,current_buffer_index);
            Next_pointer_chain(frequency_index,Previous_pointer_chain(frequency_index,current_buffer_index))=Next_pointer_chain(frequency_index,current_buffer_index);
            median_pointer(frequency_index)=Previous_pointer_chain(frequency_index,median_pointer(frequency_index));
            [Next_pointer_chain,Previous_pointer_chain]=search_and_replace_down(median_pointer(frequency_index),current_buffer_index,Next_pointer_chain,Previous_pointer_chain,spectrum_buffer,current_spectrum_value,total_median_memory_size,frequency_index);
            %else
            % nedata is exactly in the right place - don't need to do anything
            
        end
        median_vec(frequency_index)=spectrum_buffer(frequency_index,median_pointer(frequency_index));
        continue
    end
    
    
    %% remove old ind from the list
    Previous_pointer_chain(frequency_index,Next_pointer_chain(frequency_index,current_buffer_index))=Previous_pointer_chain(frequency_index,current_buffer_index);
    Next_pointer_chain(frequency_index,Previous_pointer_chain(frequency_index,current_buffer_index))=Next_pointer_chain(frequency_index,current_buffer_index);
    
    if current_spectrum_value>=spectrum_buffer(frequency_index,median_pointer(frequency_index))
        [Next_pointer_chain,Previous_pointer_chain]=search_and_replace_up(median_pointer(frequency_index),current_buffer_index,Next_pointer_chain,Previous_pointer_chain,spectrum_buffer,current_spectrum_value,total_median_memory_size,frequency_index);
    else
        [Next_pointer_chain,Previous_pointer_chain]=search_and_replace_down(median_pointer(frequency_index),current_buffer_index,Next_pointer_chain,Previous_pointer_chain,spectrum_buffer,current_spectrum_value,total_median_memory_size,frequency_index);
    end
    
    
    %% update the median pointer
    % if both old val and new val are smaller than median - do nothing
    % if both old val and new val are bigger than median - do nothing
    
    % if xold was smaller the median and newdata is bigger than median
    % increase median ptr to next value
    if previous_spectrum_value<=old_median_value && current_spectrum_value>old_median_value
        median_pointer(frequency_index)=Next_pointer_chain(frequency_index,median_pointer(frequency_index));
        median_vec(frequency_index)=spectrum_buffer(frequency_index,median_pointer(frequency_index));
        
    elseif previous_spectrum_value>=old_median_value && current_spectrum_value<old_median_value
        % if xold was bigger the median and newdata is bigger than median
        % decrease median ptr to previous value
        median_pointer(frequency_index)=Previous_pointer_chain(frequency_index,median_pointer(frequency_index));
        median_vec(frequency_index)=spectrum_buffer(frequency_index,median_pointer(frequency_index));
    else
        median_vec(frequency_index)=old_median_value;
    end
    
    %Update Min and Max vecs:
    min_vec(frequency_index) = spectrum_buffer(frequency_index,Next_pointer_chain(frequency_index,total_median_memory_size));
    max_vec(frequency_index) = spectrum_buffer(frequency_index,Previous_pointer_chain(frequency_index,total_median_memory_size));
    
end





function [N,P]=search_and_replace_up(ra,ind,N,P,x,newdata,rl_rs,jj)
while  x(jj,N(jj,ra))<newdata
    ra=N(jj,ra);
    if (N(jj,ra)==rl_rs)
        break
    end
end
%% chain in the new data
N(jj,ind)=N(jj,ra);
P(jj,N(jj,ra))=ind;
N(jj,ra)=ind;
P(jj,ind)=ra;

function [N,P]=search_and_replace_down(ra,ind,N,P,x,newdata,rl_rs,jj)
while  x(jj,P(jj,ra))>newdata
    ra=P(jj,ra);
    if (P(jj,ra)==rl_rs)
        break
    end
end
%% chain in the new data
P(jj,ind)=P(jj,ra);
N(jj,P(jj,ra))=ind;
P(jj,ra)=ind;
N(jj,ind)=ra;



