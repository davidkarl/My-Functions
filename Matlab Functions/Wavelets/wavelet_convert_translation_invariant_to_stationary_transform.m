function StatWT = wavelet_convert_translation_invariant_to_stationary_transform(translation_invariant_table)
% TI2Stat -- Convert Translation-Invariant Transform to Stationary Wavelet Transform
%  Usage
%    StatWT = TI2Stat(TIWT)
%  Inputs
%    TIWT     translation invariant table from FWT_TI
%  Outputs
%    StatWT   stationary wavelet transform table table as FWT_Stat
%
%  See Also
%    Stat2TI, FWT_TI, FWT_Stat
%
StatWT = translation_invariant_table;
[n,D1] = size(StatWT);
D = D1-1;
J = log2(n);
L = J-D;
%
index = 1;

for d=1:D,
    nb = 2^d;
    nk = n/nb;

    index = [ (index+nb/2); index];
    index = index(:)';

    for b= 0:(nb-1),
        StatWT(d*n + (index(b+1):nb:n)) = translation_invariant_table(d*n + wavelet_packet_table_indexing(d,b,n));
    end
end

for b = 0:(nb-1),
    StatWT((index(b+1):nb:n)) = translation_invariant_table(wavelet_packet_table_indexing(d,b,n));
end

%
% Copyright (c) 1994. Shaobing Chen
%