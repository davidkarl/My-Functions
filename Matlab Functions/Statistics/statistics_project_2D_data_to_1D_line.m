% Example 6.1
% Computational Statistics Handbook with MATLAB, 2nd Edition
% Wendy L. and Angel R. Martinez

% Show how to project data onto a line. 

% Specify some data. 
X = [-2 4; 2 4;6 1;8 10;7 5;11 8];
% Create the projection matrix.
theta = pi/4;
c2 = cos(theta)^2;
cs = cos(theta)*sin(theta);
s2 = sin(theta)^2;
P = [cos(theta)^2          , cos(theta)*sin(theta) ; ...
     cos(theta)*sin(theta) , sin(theta)^2                ];
 
%Now project the data onto the line:
Xp = X*P;


% Create the plot shown in Figure 6.1.
plot(Xp(:,1),Xp(:,2),'d',X(:,1),X(:,2),'o')
hold on
plot([0  12 ], [0 12 ], ':')
hold off
axis([-4 12 0 12])


