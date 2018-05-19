% Approximate chi-square distribution
% chappr

c = zeros(1,7);
fi=1;
for i=0:6
  if i>1
      fi = fi*i;
  end
  c(i+1) = 1/((2)^i*fi);
end

t=0.1:0.1:50;
ap=1./polyval(c(7:-1:1),t);
%plot(t,rf,'g',t,ap,'b') ;
%zoom on

t=-5:0.1:5 ;
rf=exp(-(t.^2)) ;
af=cos(t*pi/2)/2+0.5 ;
plot(t,rf,'b',t,af,'g') ;

% q=4 ;
% for z=0.1:0.1:10,
%  for y=1:2:10,
%    m=z-y ; s=z/q ;
%    t=-m/s ;
%    if (t>pi/2), x=0 ;
%    else 
%      lb=max(t,-pi/2) ;
%      x=s*((pi/2-t)-((lb-t)*sin(lb)+cos(lb)))/(1-sin(lb)) ;
%    end ;
%    r(floor(z*10),y)=x ;
%  end ;
% end ;
%
% plot(r) ;
      