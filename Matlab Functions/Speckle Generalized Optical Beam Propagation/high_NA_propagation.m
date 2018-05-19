function [mat_out_linear,Uc_linear,Vc_linear] = high_NA_propagation(mat_in,lambda,spacing,z)

k=2*pi/lambda;
N=size(mat_in,1);
x=[-N/2:N/2-1]*spacing;
[x1,y1]=meshgrid(x);

delta_f=1/(N*spacing);
x_f=[-N/2:N/2-1]*delta_f;
[Xf,Yf]=meshgrid(x_f);

Uc=lambda*z*Xf./sqrt(1-lambda^2*(Xf.^2+Yf.^2)).*(lambda^2*(Xf.^2+Yf.^2)<1);
Vc=lambda*z*Yf./sqrt(1-lambda^2*(Xf.^2+Yf.^2)).*(lambda^2*(Xf.^2+Yf.^2)<1);
R2=sqrt(Uc.^2+Vc.^2+z^2);

mat_out=fftshift(fft2(ifftshift(mat_in.*exp(1i*k/(2*z).*(x1.^2+y1.^2)))));
mat_out=mat_out*1/(1i*lambda)*z.*exp(1i*k*R2)./(R2.^2);

Uc_linear=lambda*z*Xf;
Vc_linear=lambda*z*Yf;

% mat_out_linear=griddata(Uc,Vc,mat_out,Uc_linear,Vc_linear);

spline_result_lsq_2D = spap2({5,5},{5,5},{Uc(1,1:end),Vc(1:end,1)},mat_out);
mat_out_linear = fnval(spline_result_lsq_2D,{Uc_linear,Vc_linear});

%  F = TriScatteredInterp(Uc(:),Vc(:),abs(mat_out(:)));
%  mat_out_linear = F(Uc_linear,Vc_linear);






