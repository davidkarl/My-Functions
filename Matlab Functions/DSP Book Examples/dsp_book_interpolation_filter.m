x=rand(1,40);
M=4;
y=interM(x,M);
plot(y); 
hold on; 
plot(y,'xr');

scatter([1:M:length(y)],x,'fill'); 
hold off;
grid








