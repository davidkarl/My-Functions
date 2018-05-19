function [row_shift,col_shift] = return_shifts_with_fourier_sampling(mat1,mat2,accuracy)

    % First upsample by a factor of 2 to obtain initial estimate
    % Embed Fourier data in a 2x larger array
    buf1ft=fft2(mat1);
    buf2ft=fft2(mat2);
    [m,n]=size(buf1ft);
    mlarge=m*2;
    nlarge=n*2;
    CC=zeros(mlarge,nlarge);
    CC(m+1-fix(m/2):m+1+fix((m-1)/2),n+1-fix(n/2):n+1+fix((n-1)/2)) = fftshift(buf1ft).*conj(fftshift(buf2ft));
  
    % Compute crosscorrelation and locate the peak 
    CC = ifft2(ifftshift(CC)); % Calculate cross-correlation
    [max1,loc1] = max(CC);
    [max2,loc2] = max(max1);
    rloc=loc1(loc2);cloc=loc2;
    CCmax=CC(rloc,cloc);
    
    % Obtain shift in original pixel grid from the position of the
    % crosscorrelation peak 
    [m,n] = size(CC); md2 = fix(m/2); nd2 = fix(n/2);
    if rloc > md2 
        row_shift = rloc - m - 1;
    else
        row_shift = rloc - 1;
    end
    if cloc > nd2
        col_shift = cloc - n - 1;
    else
        col_shift = cloc - 1;
    end
    row_shift=row_shift/2; %we devide by two because in the beginning we upsampled the matrices by 2!
    col_shift=col_shift/2;

        %%% DFT computation %%%
        % Initial shift estimate in upsampled grid
        row_shift = round(row_shift*accuracy)/accuracy; 
        col_shift = round(col_shift*accuracy)/accuracy;     
        dftshift = fix(ceil(accuracy*1.5)/2); %% Center of output array at dftshift+1-?????? why 1.5?!?!?!?!?
        % Matrix multiply DFT around the current shift estimate
        CC = conj(dftups(buf2ft.*conj(buf1ft),ceil(accuracy*1.5),ceil(accuracy*1.5),accuracy,...
            dftshift-row_shift*accuracy,dftshift-col_shift*accuracy))/(md2*nd2*accuracy^2);
        % Locate maximum and map back to original pixel grid 
        [max1,loc1] = max(CC);   
        [max2,loc2] = max(max1); 
        rloc = loc1(loc2); cloc = loc2;
        CCmax = CC(rloc,cloc);
%         rg00 = dftups(buf1ft.*conj(buf1ft),1,1,accuracy)/(md2*nd2*accuracy^2);
%         rf00 = dftups(buf2ft.*conj(buf2ft),1,1,accuracy)/(md2*nd2*accuracy^2);  
        rloc = rloc - dftshift - 1;
        cloc = cloc - dftshift - 1;
        row_shift = row_shift + rloc/accuracy;
        col_shift = col_shift + cloc/accuracy;    

    

function out=dftups(in,nor,noc,accuracy,roff,coff)
[nr,nc]=size(in);
% Compute kernels and obtain DFT by matrix products
kernc=exp((-1i*2*pi/(nc*accuracy))*( ifftshift([0:nc-1]).' - floor(nc/2) )*( [0:noc-1] - coff ));
kernr=exp((-1i*2*pi/(nr*accuracy))*( [0:nor-1].' - roff )*( ifftshift([0:nr-1]) - floor(nr/2)  ));
out=kernr*in*kernc;
return





