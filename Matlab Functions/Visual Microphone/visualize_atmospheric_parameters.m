%test different structure functions:
clear all;
clc;

%show nominal values power spectrum, wave structure and correlation
%function
kappa = linspace(0,5,4000)*10^3;
[kappa_x,kappa_y]=meshgrid(kappa);
[theta,rho_kappa]=cart2pol(kappa_x,kappa_y);
real_axis = 2*pi./kappa;

L0=50;
l0=5*10^-3;
Cn2=10^-14;
lambda=1.55*10^-6;
k=2*pi/lambda;
kappa_0=2*pi/L0;
kappa_m=5.92/l0;
delta_kappa=kappa(2)-kappa(1);
z=1000;

phi_mvk=0.033*Cn2*exp(-kappa.^2/kappa_m^2)./(kappa.^2+kappa_0^2).^(11/6);
phi_vk=0.033*Cn2./(kappa.^2+kappa_0^2).^(11/6);
phi_k=0.033*Cn2.*kappa.^(-11/3);
phi_andrews=0.033*Cn2.*(1+1.802*(kappa/kappa_m)-0.254*(kappa/kappa_m).^(7/6)).*exp(-kappa.^2/kappa_m^2)./(kappa.^2+kappa_0^2).^(11/6);
r0=(0.423*k^2*Cn2*z)^(-3/5); %r0_pw
 
r=linspace(0,10,1000)*10^-2;
D_k=6.88.*(r/r0).^(5/3);
D_vk=6.16*r0^(-5/3)*(3/5*kappa_0^(-5/3)-(r/2/kappa_0).^(5/6)/gamma(11/6).*besselk(5/6,kappa_0*r));
D_mvk=7.75*l0^(-1/3)*r0^(-5/3)*r.^2.*(1./(1+2.03/l0^2.*r.^2).^(1/6)-0.72*(kappa_0*l0)^(1/3));
corr_k=exp(-1/2*D_k);
corr_vk=exp(-1/2*D_vk);
corr_mvk=exp(-1/2*D_mvk);

% figure(1)
% % loglog(kappa,phi_k,kappa,phi_vk,kappa,phi_mvk,kappa,phi_andrews);
% loglog(real_axis,phi_k,real_axis,phi_vk,real_axis,phi_mvk,real_axis,phi_andrews);
% xlabel('[m]');
% % plot(kappa,phi_k,kappa,phi_vk,kappa,phi_mvk,kappa,phi_andrews);
% legend('kolmogorov','von kolmogorov','modified von kolmogorov','modified andrews spectra');
% title('refractive index power spectrum');
% 
% figure(2)
% plot(r,D_k,r,D_vk,r,D_mvk);
% legend('D^k','D^v^k','D^m^v^k');
% title('total phase structure funciton in the weak turbulence regime for a plane wave');
% xlabel('[m]');
% figure(3)
% plot(r,corr_k,r,corr_vk,r,corr_mvk);
% legend('k','vk','mvk');
% title('wave cross correlation function');
% xlabel('[m]');



%CHECK THE CHANGE IN THE IMPORTANT FUNCTIONS WITH CHANGING INNER AND OUTER SCALES

% %check influence of outer scale:
% L0=linspace(20,200,5);
% str=cell(length(L0),1);
% for m=1:length(L0)
%     str{m}=num2str(L0(m));
% end
% l0=5*10^-3;
% Cn2=10^-14;
% lambda=1.55*10^-6;
% k=2*pi/lambda;
% kappa_m=5.92/l0;
% delta_kappa=kappa(2)-kappa(1);
% z=1000;
% r0=(0.423*k^2*Cn2*z)^(-3/5); %r0_pw
% list=hsv(length(L0));
% figure(4)
% for m=1:length(L0)
% L0_current=L0(m);
% kappa_0=2*pi/L0_current;
% D_mvk=7.75*l0^(-1/3)*r0^(-5/3)*r.^2.*(1./(1+2.03/l0^2.*r.^2).^(1/6)-0.72*(kappa_0*l0)^(1/3));
% corr_mvk=exp(-1/2*D_mvk);
% plot(r,corr_mvk,'col',list(m,:));
% hold on;
% end
% legend(str);
% title('wave correlation function for different L0');
% xlabel('[m]');


% %check influence of inner scale:
% l0=linspace(eps,20,5)*10^-3;
% str=cell(length(l0),1);
% for m=1:length(l0)
%     str{m}=num2str(l0(m));
% end
% L0=50;
% Cn2=10^-14;
% lambda=1.55*10^-6;
% k=2*pi/lambda;
% kappa_0=2*pi/L0;
% delta_kappa=kappa(2)-kappa(1);
% z=1000;
% r0=(0.423*k^2*Cn2*z)^(-3/5); %r0_pw
% list=hsv(length(l0));
% figure(5)
% for m=1:length(l0)
% l0_current=l0(m);
% kappa_m=5.92/l0_current;
% D_mvk=7.75*l0_current^(-1/3)*r0^(-5/3)*r.^2.*(1./(1+2.03/l0_current^2.*r.^2).^(1/6)-0.72*(kappa_0*l0_current)^(1/3));
% corr_mvk=exp(-1/2*D_mvk);
% plot(r,corr_mvk,'col',list(m,:));
% hold on;
% end
% legend(str);
% title('wave correlation function for different l0');
% xlabel('[m]');


%check influence of Cn2:
Cn2=logspace(10^-17,10^-13,5);
str=cell(length(Cn2),1);
for m=1:length(l0) 
    str{m}=num2str(Cn2(m));
end
l0=5*10^-3;
L0=50;
lambda=1.55*10^-6;
k=2*pi/lambda;
kappa_0=2*pi/L0;
kappa_m=5.92/l0;
delta_kappa=kappa(2)-kappa(1);
z=1000;
r0=(0.423*k^2*Cn2*z)^(-3/5); %r0_pw
list=hsv(length(l0));
figure(6)
for m=1:length(Cn2)
r0=(0.423*k^2*Cn2_current*z)^(-3/5); %r0_pw
D_mvk=7.75*l0^(-1/3)*r0^(-5/3)*r.^2.*(1./(1+2.03/l0^2.*r.^2).^(1/6)-0.72*(kappa_0*l0)^(1/3));
corr_mvk=exp(-1/2*D_mvk);
plot(r,corr_mvk,'col',list(m,:));
hold on;
end
legend(str);
title('wave correlation function for different l0');
xlabel('[m]');





