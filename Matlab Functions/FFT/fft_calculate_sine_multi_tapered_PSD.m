function [multi_tapered_PSD] = fft_calculate_sine_multi_tapered_PSD( input_signal, sine_tapers )

%Initialize variables:
[number_of_sine_tapers , individual_sine_taper_length] = size(sine_tapers);
x_tapered_spectrum = zeros(number_of_sine_tapers , individual_sine_taper_length);
input_signal = make_column(input_signal);
input_signal_number_of_samples = length(input_signal);

%Build a Hankel matrix from input_signal according to sine taper length:
%(a hankel matrix is a square, symmetric, and are constant along anti-diagonals)
%(for us, basically the constructed hankel matrix is a matrix where the first column is the first part 
%of the signal and each column down clicking by one sample)
constructed_data_matrix_number_of_columns = input_signal_number_of_samples-individual_sine_taper_length+1;
data_hankel_matrix = hankel( input_signal(1:individual_sine_taper_length) , input_signal(individual_sine_taper_length : end) );

%Initialize multi-tapers PSD matrix for later averaging: 
multi_tapers_PSD_matrix = zeros( individual_sine_taper_length , input_signal_number_of_samples-individual_sine_taper_length+1 );

%Loop over constracted Hankel matrix columns and construct the multi-tapered PSD matrix:
for column_counter = 1:constructed_data_matrix_number_of_columns
    %get current data vector:
    current_data_vector = data_hankel_matrix(:,column_counter);
    
    %Loop over sine tapers and taper the data matrix:
    for index = 1:number_of_sine_tapers
        %get current sine taper:
        current_sine_taper = sine_tapers(index,:);
        
        %taper signal with current sine taper:
        x_tapered = current_sine_taper .* current_data_vector';
        x_tapered_fft = fft(x_tapered);
        x_tapered_spectrum(index,:) = abs(x_tapered_fft).^2;
    end
    
    %Average over the different spectrums from different sine tapers:
    multi_tapers_PSD_matrix(:,column_counter) = mean(x_tapered_spectrum,1);
end

%Average over all the different segments (seperated by one sample):
multi_tapered_PSD = mean( multi_tapers_PSD_matrix, 2);