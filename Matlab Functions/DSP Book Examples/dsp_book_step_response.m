%DSP book test step response:
impulse_response_length = 500;
a_first_order_recorsive_pole = [1/2,3/4,7/8];
number_of_poles_to_test = length(a_first_order_recorsive_pole);
a_first_order_recorsive_pole = ones(impulse_response_length,1)*a_first_order_recorsive_pole;
one_to_N_vecs_for_impulse_response = [0:impulse_response_length-1]' * ones(1,number_of_poles_to_test);

%step response is the sum of the impulse response, and in the case of
%H(z)=1/(1-az^-1) -> h(n)=a^n(n>0) -> h_step(n)=sum_up_to_n(h(n)).
%what we do here is truncate it to make it an FIR filter defined over 0<=n<=impulse_response_length
%in this case we simulate the step response by inserting ones to the filter
%and knowing that it assumes zeros before that
step_response = a_first_order_recorsive_pole .^ one_to_N_vecs_for_impulse_response;
step_responses_normalization = sum(step_response);

%the impulse responses are normalized in order to compare rise times:
impulse_response1 = step_response(:,1)/step_responses_normalization(1);
impulse_response2 = step_response(:,2)/step_responses_normalization(2);
impulse_response3 = step_response(:,3)/step_responses_normalization(3);

%response's length:
L_response = 30; 
tps = [0:L_response-1];
x = ones(L_response,1);

%responses with null initial conditions:
y = [ filter(impulse_response1,1,x) , filter(impulse_response2,1,x) , filter(impulse_response3,1,x) ];

%plot:
figure
plot(tps,y,'-',tps,y,'o');
set(gca,'ylim',[0,1.1]); 
grid


