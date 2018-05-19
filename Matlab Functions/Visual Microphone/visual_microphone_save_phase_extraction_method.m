%VISUAL MICROPHONE SAVE DIFFERENT PHASE EXTRACTION METHODS

% PHASEAMPLIFY(VIDFILE, MAGPHASE, FL, FH, FS, OUTDIR, VARARGIN)
%
% Takes input VIDFILE and motion magnifies the motions that are within a
% passband of FL to FH Hz by MAGPHASE times. FS is the videos sampling rate
% and OUTDIR is the output directory.
%
% Optional arguments:
% attenuateOtherFrequencies (false)
%   - Whether to attenuate frequencies in the stopband
% pyrType                   ('halfOctave')
%   - Spatial representation to use (see paper)
% sigma                     (0)
%   - Amount of spatial smoothing (in px) to apply to phases
% temporalFilter            (FIRWindowBP)
%   - What temporal filter to use
% 

%Get input variables:
video_file = 'Chips1-2200Hz-Mary_Had-input.avi';
output_directory = 'C:\Users\master\Desktop\matlab';
low_cutoff_frequency = 100;
high_cutoff_frequency = 1000;
Fs = 2200;
phase_magnification_factor = 15;    











