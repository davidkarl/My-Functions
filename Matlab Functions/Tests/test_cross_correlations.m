%test cross correlations:
N=512;
ROI=100;
speckle_size=10;
shiftx=1.43;
shifty=1.32;
a=create_speckles_of_certain_size_in_pixels(speckle_size,N,1,0);
b=shift_matrix(a,1,shiftx,shifty);
a=abs(a).^2;
b=abs(b).^2;
a=a(1:ROI,1:ROI);
b=b(1:ROI,1:ROI);

%corr2_ft:
corr2_ft_mat=corr2_ft(a,b,1);
%corr2_ft subtracted and normalized:
corr2_ft_normalized_mat=corr2_ft((a-mean(a(:)))/norm(a),(b-mean(b(:)))/norm(b),1);
%corr2_padded_ft:
corr2_padded_ft_mat=corr2_padded_ft(a,b,1);
%corr2_padded_ft subtracted and normalized:
corr2_padded_ft_normalized_mat=corr2_padded_ft((a-mean(a(:)))/norm(a),(b-mean(b(:)))/norm(b),1);
%cross_correlation_ft:
cross_correlation_ft_mat = cross_correlation_ft(a,b,1);
%cross_correlation_ft subtracted and normalized:
cross_correlation_ft_normalized_mat = cross_correlation_ft((a-mean(a(:)))/norm(a),(b-mean(b(:)))/norm(b),1);
%xcorr2:
xcorr2_mat=xcorr2(a,b);
%xcorr2 subtracted and normalized:
xcorr2_normalized_mat=xcorr2((a-mean(a(:)))/norm(a),(b-mean(b(:)))/norm(b));
%normxcorr2:
normxcorr2_mat=normxcorr2(a,b);
%calculate_properly_subtracted_cross_correlation_cells:
cross_correlation_dan_mat=cross_correlation_dan_find_max(a,b,1);

%get central 3X3 elements:
%(1). un-normalized:
corr2_ft_mat = get_sub_matrix_around_max(corr2_ft_mat,1);
corr2_padded_ft_mat = get_sub_matrix_around_max(corr2_padded_ft_mat,1);
xcorr2_mat = get_sub_matrix_around_max(xcorr2_mat,1);
xcorr2_mat = rot90(xcorr2_mat,2);
cross_correlation_ft_mat = get_sub_matrix_around_max(cross_correlation_ft_mat,1);
%(2). normalized:
corr2_ft_normalized_mat = get_sub_matrix_around_max(corr2_ft_normalized_mat,1);
corr2_padded_ft_normalized_mat = get_sub_matrix_around_max(corr2_padded_ft_normalized_mat,1);
xcorr2_normalized_mat = get_sub_matrix_around_max(xcorr2_normalized_mat,1);
normxcorr2_mat = get_sub_matrix_around_max(normxcorr2_mat,1);
xcorr2_normalized_mat = rot90(xcorr2_normalized_mat,2);
cross_correlation_dan_mat = rot90(cross_correlation_dan_mat,1); 
cross_correlation_ft_normalized_mat = get_sub_matrix_around_max(cross_correlation_ft_normalized_mat,1);

%show differences:
%(1). un-normalized cross correlations:
figure(1)
imagesc(corr2_ft_mat-corr2_padded_ft_mat); %not exactly the same, corr2_padded_ft may be considered as more correct
colorbar;
title('corr2_ft vs. corr2_padded_ft');
figure(2)
imagesc(xcorr2_mat - corr2_ft_mat); %not exactly the same because xcorr2 apperantly uses padding as can be concluded by it resemblence to corr2_padded_ft
colorbar;
title('xcorr2 vs. corr2_ft');
figure(3)
imagesc(xcorr2_mat - corr2_padded_ft_mat); %practically the same, xcorr2=real space, corr2_padded_ft=frequency space, it appears xcorr2 is using padding
colorbar;
title('xcorr2 vs. corr2_padded_ft');
figure(4)
imagesc(xcorr2_mat - cross_correlation_ft_mat);
colorbar;
title('xcorr2 vs. cross_correlation_ft');
%(2). normalized cross correlations:
figure(5)
imagesc(xcorr2_normalized_mat-normxcorr2_mat); %normxcorr2 is normalized differently for each cross correlation cell, xcorr2 is not
colorbar;
title('normalized xcorr2 vs. normxcorr2');
figure(6)
imagesc(normxcorr2_mat-cross_correlation_dan_mat); %cross correlation dan normalized like normxcorr2 but chopps off irrelevant cells in original ROIs
colorbar;
title('normalized normxcorr2 vs. dan');
figure(7)
imagesc(xcorr2_normalized_mat-cross_correlation_dan_mat); %there is quite a disperancy because xcorr2 doesn't use individualized normalization
colorbar;
title('normalized xcorr2 vs. dan');
figure(8)
imagesc(corr2_ft_normalized_mat-cross_correlation_dan_mat); 
colorbar;
title('normalized corr2_ft vs. dan');
figure(9)
imagesc(corr2_padded_ft_normalized_mat-cross_correlation_dan_mat);
colorbar;
title('normalized corr2_padded_ft vs. dan');
figure(10)
imagesc(corr2_ft_normalized_mat-xcorr2_normalized_mat);
colorbar;
title('normalized corr2_ft vs. xcorr2');
figure(11)
imagesc(corr2_padded_ft_normalized_mat-xcorr2_normalized_mat); %exactly the same
colorbar;
title('normalized corr2_padded_ft vs. xcorr2');

%show:   
figure(1)
imagesc(corr2_ft_mat); 
colorbar;
figure(2)
imagesc(corr2_ft_normalized_mat);
colorbar;
figure(3)
imagesc(corr2_padded_ft_mat);
colorbar;
figure(4)
imagesc(corr2_padded_ft_normalized_mat);
colorbar;
figure(7)
imagesc(normxcorr2_mat);
colorbar;
figure(8)
imagesc(cross_correlation_dan_mat);
colorbar;





