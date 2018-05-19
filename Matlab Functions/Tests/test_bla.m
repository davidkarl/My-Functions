
height = 1000;
width = 1000;
x = randn(1,height*width);
tic
for i=1:10
    y = reshape(x,height,width);
end
toc


% Bivariate test
X = 1:1000;
Y = 1:1000;
[XX,YY] = meshgrid(X,Y);
W = sin(sqrt(XX.^2+YY.^2));
spline = fn2fm(spapi({aptknt(X,3),aptknt(Y,3)},{X,Y},W),'pp');
new_points = [X(:),Y(:)]';
tic;yy = ppmval(X,spline); toc;


original_mat = create_speckles_of_certain_size_in_pixels(30,1000,1,1);
x_mesh = 1:1000;
y_mesh = 1:1000;
[X,Y] = meshgrid(x_mesh);
sigma = 0.1;
displacements_x = sigma*randn(1000,1000);
displacements_y = sigma*randn(1000,1000);
X2 = X + displacements_x;
Y2 = Y + displacements_y;
new_points = [X2(:),Y2(:)]';
spline = fn2fm(spapi({aptknt(x_mesh,3),aptknt(y_mesh,3)},{x_mesh,y_mesh},original_mat),'pp');
tic
bla = interp2(X,Y,original_mat,X2,Y2,'cubic');
toc
tic
bla2 =  ppmval(new_points,spline);
toc


original_mat = abs(create_speckles_of_certain_size_in_pixels(30,1024,1,1)).^2;
tic
for i=1:10
    blabla = fft2(original_mat);
end
toc

