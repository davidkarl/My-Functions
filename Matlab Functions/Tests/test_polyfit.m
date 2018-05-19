%test polyfit:

a1 = 10;
x = linspace(0,10,1000);
y = a1*x;
b = 1;

for counter = 1:100
    noise = b*randn(1000,1)';
    y = y + noise;
    
%     fft_plot_fft(y,44100,0,1,1);
    
    P = polyfit(x,y,1);
    prediction = P(1); 
    error_vec(counter) = a1 - prediction;
end

bla = std(error_vec);
 
1;
 




