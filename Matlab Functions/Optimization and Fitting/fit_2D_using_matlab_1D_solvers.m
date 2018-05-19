function [beta,output_vec] = fit_2D_using_matlab_1D_solvers(x,y,z,solver,modelFun,initial_coeffs_vec,options,vargin)
%unite x and y column vectors:
xx = [x;y];

if strcmp(solver,'nlinfit')
    [beta,R,J,covB,MSE,ErrorModelInfo] = nlinfit(xx,z,@fitfit2,initial_coeffs_vec,options);
    output_vec{1}=R;
    output_vec{2}=J;
    output_vec{3}=covB;
    output_vec{4}=MSE;
    output_vec{5}=ErrorModelInfo;
elseif strcmp(solver,'lsqcurvefit') || strcmp(solver,'lsqnonlin') || strcmp(solver,'lsq')
    [beta,resnorm,residual,exitflag,output,lambda,jacobian] = lsqcurvefit(@fitfit2,initial_coeffs_vec,xx,z);
    output_vec{1}=resnorm;
    output_vec{2}=residual;
    output_vec{3}=exitflag;
    output_vec{4}=lambda;
    output_vec{5}=jacobian;
end

function [f] = fitfit2(coeffs_vec,combined_x_and_y)
    x_original=combined_x_and_y(1:length(combined_x_and_y)/2);
    y_original=combined_x_and_y(length(combined_x_and_y)/2+1:end);
    f=modelFun(coeffs_vec,x_original,y_original);
end


end