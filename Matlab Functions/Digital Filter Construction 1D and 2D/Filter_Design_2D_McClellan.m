


function [Filter_1D,Filter_2D]=Filter_Design_2D_McClellan(Type,edges,Ap,Aa,transformation_vector)

%
%
% Written By:   Iman Moazzen
%               Dept. of Electrical Engineering
%               University of Victoria
%                                                                                          
% [Filter_1D,Filter_2D]=Filter_Design_2D_McClellan(Type,edges,Ap,Aa,transformation_vector)
% 
% This function can be used to design a lowpass, highpass, bandpass, or bandstop
% two-dimensional filter which satisfies prescribed specifications. 
% 
% _ Type can be ‘Lowpass’, ‘Highpass’, ‘Bandpass’, or 'Bandstop'
% _ edges is a vector of normalized frequencies (rad/s) including passband and stopband edges.
%   (frequencies must be in an increasing order)
% _ Ap: peak to peak passband ripple (db)
% _ Aa: minimum stopband attenuation (db)
% _ transformation_vector: this is a vector with 4 elements which maps 1D
% space to 2D space [1]. These coefficients can be found based on the method
% presented in [2]. Some examples are as follows:
%
% 2D Filter with circularly symmetric spectrum=[-0.5 0.5 0.5 0.5];
% 2D Filter with elliptic spectrum=[-2.4973 2.9006 0.3127  0.2840];
% 2D Filter with fan shape spectrum=[0 0.5 -0.5 0];
%
% The algorithm first designs 1D filter based on kaiser method and
% Dig_Filter(http://www.mathworks.com/matlabcentral/fileexchange/30321-digfilter).
% Then by using the transformation vector and Chebyshev Polynomial 
% (http://www.mathworks.com/matlabcentral/fileexchange/4913-chebyshevpoly-m)
% 2D filter will be designed. 
%
% The amplitude response of 2D and 1D filters as well as contours of the
% transformation function will be shown at the output.
%
% [Filter_1D,Filter_2D] are 1D and 2D filters’ coefficients, respectively.
% 
% Example:
% [Filter_1D,Filter_2D]=Filter_Design_2D_McClellan('lowpass',[0.1*pi,0.2*pi],0.5,30,[-0.5 0.5 0.5 0.5]);
% Which designs a lowpass 2D FIR filter with circularly symmetric spectrum using Kaiser Method. 
% Passband edge is 0.1*pi, and stopband edge is 0.2*pi, 
% Ap=0.5 (db) and Aa=30 (db). 
%
%
% -------------------------------------------------------------------------
% References:
% [1] D.E. Dudgeon, R.M. Mersereau, "Multidimensional digital
% signal processing", Prentice-Hall.
% [2] R.M. Mersereau, W.F.G. Mecklenbrauker, T.F. Quatieri, "McClellan transformations 
% for two-dimensional digital filtering: I-Design", IEEE transactions on circuit and systems, 1976.  
%
%
%
% To find other Matlab functions about filter design, please visit
% http://www.ece.uvic.ca/~imanmoaz/index_files/index1.htm


ws=2*pi;
error_flag=0;


if nargin<5
    disp('Not enough input arguments')
    error_flag=1;
end


if nargin==5
    switch lower(Type)
           case {'lowpass'}
               if length(edges)~=2
                   disp('Lowpass filter needs exactly two edges')
                   error_flag=1;
               end
           case {'highpass'}
               if length(edges)~=2
                   disp('Highpass filter needs exactly two edges')
                   error_flag=1;
               end    
           case {'bandpass'}
               if length(edges)~=4
                   disp('Bandpass filter needs exactly four edges')
                   error_flag=1;
               end                
           case {'bandstop'}
               if length(edges)~=4
                   disp('Bandstop filter needs exactly four edges')
                   error_flag=1;
               end  
        otherwise
               disp('Unknown Type')
               error_flag=1;
    end

    if sum(sort(edges)~=edges)~=0
        disp('The cut-off frequencies must be in an increasing order')
        error_flag=1;
    end


    if length(transformation_vector)~=4
       disp('transformation vector must have exactly 4 elements')
       error_flag=1;
    end    

end

if error_flag==0;
    
    [Num,Den]=Dig_Filter('FIR',Type,'Kaiser',edges,ws,Ap,Aa); 

    wx=linspace(-ws/2,ws/2,128);
    wy=linspace(-ws/2,ws/2,128);

    [wx,wy]=meshgrid(wx,wy);


    Fp=transformation_vector(1)+transformation_vector(2)*cos(wx)+transformation_vector(3)*cos(wy)+transformation_vector(4)*cos(wx).*cos(wy);

    F_max=max(max(Fp));
    F_min=min(min(Fp));

    c1=2/(F_max-F_min);
    c2=c1*F_max-1;

    F=c1*Fp-c2;


    M=(length(Num)-1)/2;
    a=[Num(M+1) 2*Num(M+2:end)];

    [mm,nn]=size(F);


    for i=1:mm
        for j=1:nn

            temp=0;

            for n=0:M;    
                temp=a(n+1)*polyval(ChebyshevPoly(n),F(i,j))+temp;
            end
            H(i,j)=temp;     
        end
    end
    
    Filter_1D=Num;
    Filter_2D=real(ifft2(H));
    
    subplot(2,3,[1 2 4 5])
    mesh(wx,wy,20*log10(abs(H)),'linewidth',2)
    axis([-pi pi -pi pi -140 10])
    xlabel('w_1','fontsize',10)
    ylabel('w_2','fontsize',10)
    zlabel('Amplitude Response (dB)','fontsize',10)
    title('Frequency Response of 2D Filter','fontsize',10)
    
    subplot(2,3,3)
    contour(wx,wy,F,'linewidth',2);
    title('Contours of Transfer Function','fontsize',10)
    subplot(2,3,6)
    [FF,WW]=freqz(Filter_1D);
    plot(WW,20*log10(abs(FF)),'linewidth',2)
    axis([0 pi -100 10])
    xlabel('w','fontsize',10)
    ylabel('Amplitude Response (dB)','fontsize',10)
    title('Frequency Response of 1D Filter','fontsize',10)
    
else
    Filter_1D=[];
    Filter_2D=[];
    return
end







