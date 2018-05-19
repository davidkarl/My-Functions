function [steering_angles_matrix, steered_riesz_matrix_3D] = riesz_get_angles_that_give_maximum_response(...
                                                                                    riesz_matrix_3D, ...
                                                                                    riesz_transform_order, ...
                                                                                    riesz_channel_to_be_maximized, ...
                                                                                    flag_debug_mode)
%RIESZANGLE compute the angles that gives maximum response for a given Riesz channel
%
% --------------------------------------------------------------------------
% Input arguments: 
%
% ORIG the original Riesz-wavelet coefficients at a given scale
%
% ORDER the order of the Riesz transform
%
% CHANNEL Riesz channel for which the response is to be maximized
%
% DEBUG Display some computation details if 1. Optional. Default is 0;
%
% --------------------------------------------------------------------------
% Output arguments:
%
% TH matrix of angles that are estimated pointwise in the wavelet band.
%
% MX Riesz-wavelet coefficient steered with respect to the angles in TH.
%
% --------------------------------------------------------------------------
%
% Part of the Generalized Riesz-wavelet toolbox
%
% Author: Dimitri Van De Ville and Nicolas Chenouard. Ecole Polytechnique Federale de Lausanne.
%
% Version: Feb. 7, 2012

if nargin < 4,
    flag_debug_mode = 0;
end

template = zeros(1,riesz_transform_order+1);
template(riesz_channel_to_be_maximized) = 1;
[steering_angles_matrix, steered_riesz_matrix_3D] = riesz_angle_template(riesz_matrix_3D, template, riesz_transform_order, flag_debug_mode);

% % load polynomial coefficients
% matfn=sprintf('RieszAngle%d-Ch%d.dat',order,channel);
% mat=reshape(load(matfn),[order+1 order+1]);
% %mat=shiftdim(mat,1);
% 
% if debug,
%     % visualize polynomial
%     fprintf('\n');
%     for iter1=1:order+1, % degree
%         fprintf('tan(t)^%d ( ',order+1-iter1);
%         for iter2=1:order+1, % channel
%             fprintf('%+3.1f ch[%d] ',mat(iter2,iter1),iter2);
%         end;
%         fprintf(')\n');
%     end;
% end;
% 
% % load steering matrix
% steermatfn=sprintf('RieszSteer%d.dat',order);
% steermat=shiftdim(reshape(load(steermatfn),[order+1 order+1 order+1]),1);
% 
% terms=zeros(size(orig));
% for iter1=1:order+1,
%     for iter2=1:order+1,
%         if mat(iter2,iter1),
%             terms(:,:,iter1)=terms(:,:,iter1)+orig(:,:,iter2).*mat(iter2,iter1);
%         end;
%     end;
% end
% 
% th=zeros(size(terms, 1), size(terms, 2));
% mx=zeros(size(th));
% for iterx1=1:size(terms,1),
%     for iterx2=1:size(terms,2),
%         for iter=1:order+1,
%             C(iter)=terms(iterx1,iterx2, iter);
%         end;
%         R=real(roots(C(find(abs(imag(C))<1e-5))));
%         if isempty(R),
%             R= [0];
%         end;
%         tha=atan(R);
%         V = zeros(size(tha));
%         for iterch=1:order+1,       % channel
%             for iterterm=1:order+1, % term
%                 V = V + ...
%                     steermat(channel,iterch,iterterm) * ...
%                     cos(tha).^(order-iterch+1).*sin(tha).^(iterch-1) .* ...
%                     orig{iterch}(iterx1,iterx2);
%             end;
%         end;
%         idx=find(abs(V(:))==max(abs(V(:))));
%         th(iterx1,iterx2)=-tha(idx(1));
%         mx(iterx1,iterx2)=max(abs(V(:)));
%     end;
%end;