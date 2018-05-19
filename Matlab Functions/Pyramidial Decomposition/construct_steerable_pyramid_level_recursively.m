function [pyr,pind] = construct_steerable_pyramid_level_recursively(lo0,height,lofilt,bfilts,edges)
% [PYR, INDICES] = buildSpyrLevs(LOIM, HEIGHT, LOFILT, BFILTS, EDGES)
%
% Recursive function for constructing levels of a steerable pyramid.  This
% is called by buildSpyr, and is not usually called directly.


if (height <= 0)

  pyr = lo0(:);
  pind = size(lo0);

else

  % Assume square filters:
  bfiltsz =  round(sqrt(size(bfilts,1)));
 
  bands = zeros(numel(lo0),size(bfilts,2));
  bind = zeros(size(bfilts,2),2);

  for b = 1:size(bfilts,2)
    filt = reshape(bfilts(:,b),bfiltsz,bfiltsz);
    band = corr2_downsample(lo0, filt, edges);
    bands(:,b) = band(:);
    bind(b,:)  = size(band);
  end
	
  lo = corr2_downsample(lo0, lofilt, edges, [2 2], [1 1]);
  
  [npyr,nind] = construct_steerable_pyramid_level_recursively(lo, height-1, lofilt, bfilts, edges);

  pyr = [bands(:); npyr];
  pind = [bind; nind];
	
end
