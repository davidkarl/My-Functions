%test integrate and dump 

%Create an integrate and dump filter having an integration period of 20 samples.
integrate_and_dump_filter = comm.IntegrateAndDumpFilter(20);

%Generate binary data:
d = randi([0 1],50,1);

%Upsample the data, and pass it through an AWGN channel:
x = upsample(d,20);
y = awgn(x,25,'measured');

%Pass the noisy data through the filter:
z = step(integrate_and_dump_filter,y);

%Plot the original and filtered data. The integrate and dump filter removes most of the noise effects.
stairs([d z])
legend('Original Data','Filtered Data')
xlabel('Samples')
ylabel('Amplitude')
grid



