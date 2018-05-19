%dsp book get AR2 coefficients from resonance frequency:

rho_poles_modulai = [0.98:0.001:0.999];
F_resonance_frequency = 0.1;
expfR = exp(-2*pi*1i*F_resonance_frequency);

%if the poles of the AR2 are complex conjugates, therefore we get that the
%transfer function H(z) can be written as H(z)=1/(1-p1*z^-1)(1-p2*z^-1) =
%1/(1-rho*exp(i*theta)*z^-1)(1-rho*exp(-i*theta)*z^-1) and we get the
%following results for a1,a2 by comparing the above expression to
%H(Z)=1/(1+a1*z^-1+a2*z^-2)
a2 = rho_poles_modulai.^2;
a1 = -4*a2*cos(2*pi*F_resonance_frequency)./(1+a2);

%now we compute the gain at the resonance frequency:
gain_function_inverse = 1+expfR*a1 + expfR^2*a2;
gain_function_inverse_modulus = abs(gain_function_inverse);
gain_function_inverse_phase =angle(gain_function_inverse);

%make input signal at resonance frequency:
number_of_samples=4000;
t_vec=[0:number_of_samples-1];
x=sin(2*pi*F_resonance_frequency*t_vec);

%go over the different pole modului and see that getting close to 1 gives a
%longer transient response and a larger gain
figure(1);
for k=1:length(rho_poles_modulai) 
   AA = [1,a1(k),a2(k)];
   set(gca,'ylim',[-1.1,1.1],'xlim',[0,30]);
   xe=x;
   y=filter(1,AA,xe);
   plot(t_vec,y);
   grid;
   title(sprintf('rho=%5.3f',rho_poles_modulai(k)));
%    set(gca,'ylim',[-1.1,1.1]);
   pause(0.1);
end





