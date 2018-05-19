
%create arduino object and connect to board
if exist('arduino_board','var') && isa(arduino_board,'arduino') && isvalid(arduino_board),
    % nothing to do    
else
    arduino_board = arduino('COM10');
end

%DELETE ARDUINO OBJECT AT END OF WORK:
% delete(instrfind({'Port'},{'COM10'}));

 
pinMode(arduino_board,9,'output');
digitalWrite(arduino_board,9,1);
pause(1);
digitalWrite(arduino_board,9,0);
pause(1);
digitalWrite(arduino_board,9,1);


