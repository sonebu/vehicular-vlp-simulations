clear all; close all; clc
addpath('fcn')

%% Inputs

% Simulation Parameters
t_stop      = 1e0;                 % [s]
t_dt        = 1e-3;                % [s]
t           = [t_dt:t_dt:t_stop];  % [s]
vehicle.t.values    = t(:);        % [s]
vehicle.t.start     = 0;           % [s]
vehicle.t.stop      = t_stop;      % [s]
vehicle.t.dt        = t_dt;        % [s]

% Vehicle dimensions (hardcoded)
vehicle.ego.width        = 1.6; 
vehicle.target.width     = 1.6;
vehicle.ego.length       = 5.0;
vehicle.target.length    = 5.0;

% Define waypoints, rxv absolute, txv relative to rxv, waypoints are for
% front center bumper in [m]

%%% Becha scenario
% t_stop      = 1e1;                 % [s]
% t_dt        = 5e-3;                % [s]
% t           = [t_dt:t_dt:t_stop];  % [s]
% vehicle.t.values    = t(:);        % [s]
% vehicle.t.start     = 0;           % [s]
% vehicle.t.stop      = t_stop;      % [s]
% vehicle.t.dt        = t_dt;        % [s]
% target_front_center_x_relative  = [0 3.5];
% target_front_center_y_relative  = [6 6] + vehicle.target.length;
% ego_front_center_x_absolute     = [0 0];
% ego_front_center_y_absolute     = [0 1000];

%%% Join Platoon
target_front_center_x_relative  = [-2 0.2 0.2 1 4 10];
target_front_center_y_relative  = [6 8 6 5 4 1] + vehicle.target.length;
ego_front_center_x_absolute     = [0 0];
ego_front_center_y_absolute     = [0 60];

%%% Lane Change
% target_front_center_x_relative  = [-2 0 2 ];
% target_front_center_y_relative  = [6 3  6] + vehicle.target.length;
% ego_front_center_x_absolute     = [0 0];
% ego_front_center_y_absolute     = [0 60];

% Interpolate method for waypoints
% be careful with this, pchip can generate physically meaningless 
% vehicle movements, check with test.avi at v2lcDataGen.
interp_method = 'spline'; 

%% Simulation. 

% Interpolate the waypoints to size of the vehicle time array
tmp                             = linspace(1,length(target_front_center_x_relative),length(t));
target_front_center_x_relative  = interp1(1:length(target_front_center_x_relative),target_front_center_x_relative,tmp,interp_method);
target_front_center_y_relative  = interp1(1:length(target_front_center_y_relative),target_front_center_y_relative,tmp,interp_method);
tmp                             = linspace(1,length(ego_front_center_x_absolute),length(t));
ego_front_center_x_absolute     = interp1(1:length(ego_front_center_x_absolute),ego_front_center_x_absolute,tmp,interp_method);
ego_front_center_y_absolute     = interp1(1:length(ego_front_center_y_absolute),ego_front_center_y_absolute,tmp,interp_method);
clear tmp

% Compute heading: average of heading for 2 delta movements at time i 
%                  (i.e., i-1 -> i and i -> i+1), gives heading
% ASSUMPTION: linear model, i.e., Ackermann steering but small sideslip,
%             (i.e., sin(sideslip) = sideslip, see reference in transaction)
ego_heading_absolute = zeros(1,length(ego_front_center_x_absolute));
for i=2:length(ego_heading_absolute)-1
    ego_heading_absolute(i) = rad2deg( (  atan2(ego_front_center_x_absolute(i)-ego_front_center_x_absolute(i-1),...
                                                ego_front_center_y_absolute(i)-ego_front_center_y_absolute(i-1)) ...
                                        + atan2(ego_front_center_x_absolute(i+1)-ego_front_center_x_absolute(i),...
                                                ego_front_center_y_absolute(i+1)-ego_front_center_y_absolute(i))     )/2 );
end
% this heading computation method is undefined for the first sample since
% it's a delta method -> just set 2 to 1 to avoid jumps.
ego_heading_absolute(1) = ego_heading_absolute(2); ego_heading_absolute(end) = ego_heading_absolute(end-1);

% Compute absolute target coordinates using ego absolute + target relative
target_front_center_x_absolute = ego_front_center_x_absolute + target_front_center_x_relative;
target_front_center_y_absolute = ego_front_center_y_absolute + target_front_center_y_relative;

clear tmp

% Compute heading: average of heading for 2 delta movements at time i 
%                  (i.e., i-1 -> i and i -> i+1), gives heading
% ASSUMPTION: linear model, i.e., Ackermann steering but small sideslip,
%             (i.e., sin(sideslip) = sideslip, see reference in transaction)
target_heading_absolute = zeros(1,length(target_front_center_x_absolute));
for i=2:length(ego_heading_absolute)-1
    target_heading_absolute(i) = rad2deg( (atan2(target_front_center_x_absolute(i)-target_front_center_x_absolute(i-1),...
                                                 target_front_center_y_absolute(i)-target_front_center_y_absolute(i-1))...
                                         + atan2(target_front_center_x_absolute(i+1)-target_front_center_x_absolute(i),...
                                                 target_front_center_y_absolute(i+1)-target_front_center_y_absolute(i))    )/2 );
end
% this heading computation method is undefined for the first and last sample 
% since it's a delta method -> just set 2 to 1 to avoid jumps.
target_heading_absolute(1) = target_heading_absolute(2); target_heading_absolute(end) = target_heading_absolute(end-1);

% Compute speed: this is actually GPS-like speed on the front bumper center,
%                not "odometric", so it might be different than tachograph 
%                values due to wheel slip, but since we're not making any 
%                detailed assumptions on vehicle sensors (which may obtain 
%                this GPS-like value after corrections from accelerometers)
%                we're using this speed as is in the AoA1 solution.
%                Computation is "trapezoidal", like the heading method
ego_front_center_speed_absolute = zeros(1,length(ego_front_center_x_absolute));
target_front_center_speed_absolute = zeros(1,length(target_front_center_x_absolute));
for i=2:length(ego_front_center_speed_absolute)-1
    ego_front_center_speed_absolute(i)    =  (   sqrt( (ego_front_center_x_absolute(i)-ego_front_center_x_absolute(i-1)).^2 ...
                                                      +(ego_front_center_y_absolute(i)-ego_front_center_y_absolute(i-1)).^2  )... 
                                               + sqrt( (ego_front_center_x_absolute(i+1)-ego_front_center_x_absolute(i)).^2 ...
                                                      +(ego_front_center_y_absolute(i+1)-ego_front_center_y_absolute(i)).^2  ) ...
                                             )/(2*t_dt);
    target_front_center_speed_absolute(i) = (    sqrt( (target_front_center_x_absolute(i)-target_front_center_x_absolute(i-1)).^2 ...
                                                      +(target_front_center_y_absolute(i)-target_front_center_y_absolute(i-1)).^2  )... 
                                               + sqrt( (target_front_center_x_absolute(i+1)-target_front_center_x_absolute(i)).^2 ...
                                                      +(target_front_center_y_absolute(i+1)-target_front_center_y_absolute(i)).^2  )...
                                            )/(2*t_dt);
end
% this speed computation method is undefined for the first and last sample 
% since it's a delta method -> just set 2 to 1 to avoid jumps.
ego_front_center_speed_absolute(1)     = ego_front_center_speed_absolute(2);    ego_front_center_speed_absolute(end)    = ego_front_center_speed_absolute(end-1);
target_front_center_speed_absolute(1)  = target_front_center_speed_absolute(2); target_front_center_speed_absolute(end) = target_front_center_speed_absolute(end-1);

% Save output for front center absolutes
vehicle.target.front_center.x       = target_front_center_x_absolute(:);
vehicle.target.front_center.y       = target_front_center_y_absolute(:);
vehicle.target.front_center.speed   = target_front_center_speed_absolute(:);
vehicle.target.heading              = target_heading_absolute(:);
vehicle.ego.front_center.x          = ego_front_center_x_absolute(:);
vehicle.ego.front_center.y          = ego_front_center_y_absolute(:);
vehicle.ego.front_center.speed      = ego_front_center_speed_absolute(:);
vehicle.ego.heading                 = ego_heading_absolute(:);
clearvars -except vehicle

% Compute positions for 
%  - front TXs (4 left, 3 right)
%  - rear TXs  (1 left, 2 right)
%  - rear center
%  based on front center positions and heading
target_rear_center_x_absolute = zeros(1, length(vehicle.t.values));
target_rear_center_y_absolute = zeros(1, length(vehicle.t.values));
target_tx1_qrx4_x_absolute = zeros(1, length(vehicle.t.values));
target_tx2_qrx3_x_absolute = zeros(1, length(vehicle.t.values));
target_tx3_qrx2_x_absolute = zeros(1, length(vehicle.t.values));
target_tx4_qrx1_x_absolute = zeros(1, length(vehicle.t.values));
target_tx1_qrx4_y_absolute = zeros(1, length(vehicle.t.values));
target_tx2_qrx3_y_absolute = zeros(1, length(vehicle.t.values));
target_tx3_qrx2_y_absolute = zeros(1, length(vehicle.t.values));
target_tx4_qrx1_y_absolute = zeros(1, length(vehicle.t.values));
ego_rear_center_x_absolute = zeros(1, length(vehicle.t.values));
ego_rear_center_y_absolute = zeros(1, length(vehicle.t.values));
ego_tx1_qrx4_x_absolute = zeros(1, length(vehicle.t.values));
ego_tx2_qrx3_x_absolute = zeros(1, length(vehicle.t.values));
ego_tx3_qrx2_x_absolute = zeros(1, length(vehicle.t.values));
ego_tx4_qrx1_x_absolute = zeros(1, length(vehicle.t.values));
ego_tx1_qrx4_y_absolute = zeros(1, length(vehicle.t.values));
ego_tx2_qrx3_y_absolute = zeros(1, length(vehicle.t.values));
ego_tx3_qrx2_y_absolute = zeros(1, length(vehicle.t.values));
ego_tx4_qrx1_y_absolute = zeros(1, length(vehicle.t.values));
for i=1:length(vehicle.t.values)
    %%%%%% Target %%%%%%
    % front center to left head light
    target_tx4_qrx1_x_absolute(i) = vehicle.target.front_center.x(i) - (vehicle.target.width/2)*cosd(vehicle.target.heading(i));
    target_tx4_qrx1_y_absolute(i) = vehicle.target.front_center.y(i) + (vehicle.target.width/2)*sind(vehicle.target.heading(i));
    
    % front center to right head light
    target_tx3_qrx2_x_absolute(i) = vehicle.target.front_center.x(i) + (vehicle.target.width/2)*cosd(vehicle.target.heading(i));
    target_tx3_qrx2_y_absolute(i) = vehicle.target.front_center.y(i) - (vehicle.target.width/2)*sind(vehicle.target.heading(i));

    % front center to rear center
    target_rear_center_x_absolute(i) = vehicle.target.front_center.x(i) - vehicle.target.length*sind(vehicle.target.heading(i));
    target_rear_center_y_absolute(i) = vehicle.target.front_center.y(i) - vehicle.target.length*cosd(vehicle.target.heading(i));
    
    % rear center to left tail light
    target_tx1_qrx4_x_absolute(i) = target_rear_center_x_absolute(i) - (vehicle.target.width/2)*cosd(vehicle.target.heading(i));
    target_tx1_qrx4_y_absolute(i) = target_rear_center_y_absolute(i) + (vehicle.target.width/2)*sind(vehicle.target.heading(i));

    % rear center to right tail light
    target_tx2_qrx3_x_absolute(i) = target_rear_center_x_absolute(i) + (vehicle.target.width/2)*cosd(vehicle.target.heading(i));
    target_tx2_qrx3_y_absolute(i) = target_rear_center_y_absolute(i) - (vehicle.target.width/2)*sind(vehicle.target.heading(i));

    %%%%%% Ego %%%%%%   
    % front center to left head light
    ego_tx4_qrx1_x_absolute(i) = vehicle.ego.front_center.x(i) - (vehicle.ego.width/2)*cosd(vehicle.ego.heading(i));
    ego_tx4_qrx1_y_absolute(i) = vehicle.ego.front_center.y(i) + (vehicle.ego.width/2)*sind(vehicle.ego.heading(i));
    
    % front center to right head light
    ego_tx3_qrx2_x_absolute(i) = vehicle.ego.front_center.x(i) + (vehicle.ego.width/2)*cosd(vehicle.ego.heading(i));
    ego_tx3_qrx2_y_absolute(i) = vehicle.ego.front_center.y(i) - (vehicle.ego.width/2)*sind(vehicle.ego.heading(i));

    % front center to rear center
    ego_rear_center_x_absolute(i) = vehicle.ego.front_center.x(i) - vehicle.ego.length*sind(vehicle.ego.heading(i));
    ego_rear_center_y_absolute(i) = vehicle.ego.front_center.y(i) - vehicle.ego.length*cosd(vehicle.ego.heading(i));
    
    % rear center to left tail light
    ego_tx1_qrx4_x_absolute(i) = ego_rear_center_x_absolute(i) - (vehicle.ego.width/2)*cosd(vehicle.ego.heading(i));
    ego_tx1_qrx4_y_absolute(i) = ego_rear_center_y_absolute(i) + (vehicle.ego.width/2)*sind(vehicle.ego.heading(i));

    % rear center to right tail light
    ego_tx2_qrx3_x_absolute(i) = ego_rear_center_x_absolute(i) + (vehicle.ego.width/2)*cosd(vehicle.ego.heading(i));
    ego_tx2_qrx3_y_absolute(i) = ego_rear_center_y_absolute(i) - (vehicle.ego.width/2)*sind(vehicle.ego.heading(i));
end

% Save output for rest of the absolutes
vehicle.target.rear_center.x = target_rear_center_x_absolute(:);
vehicle.target.rear_center.y = target_rear_center_y_absolute(:);
vehicle.target.tx1_qrx4.x    = target_tx1_qrx4_x_absolute(:);
vehicle.target.tx1_qrx4.y    = target_tx1_qrx4_y_absolute(:);
vehicle.target.tx2_qrx3.x    = target_tx2_qrx3_x_absolute(:);
vehicle.target.tx2_qrx3.y    = target_tx2_qrx3_y_absolute(:);
vehicle.target.tx3_qrx2.x    = target_tx3_qrx2_x_absolute(:);
vehicle.target.tx3_qrx2.y    = target_tx3_qrx2_y_absolute(:);
vehicle.target.tx4_qrx1.x    = target_tx4_qrx1_x_absolute(:);
vehicle.target.tx4_qrx1.y    = target_tx4_qrx1_y_absolute(:);
vehicle.ego.rear_center.x    = ego_rear_center_x_absolute(:);
vehicle.ego.rear_center.y    = ego_rear_center_y_absolute(:);
vehicle.ego.tx1_qrx4.x       = ego_tx1_qrx4_x_absolute(:);
vehicle.ego.tx1_qrx4.y       = ego_tx1_qrx4_y_absolute(:);
vehicle.ego.tx2_qrx3.x       = ego_tx2_qrx3_x_absolute(:);
vehicle.ego.tx2_qrx3.y       = ego_tx2_qrx3_y_absolute(:);
vehicle.ego.tx3_qrx2.x       = ego_tx3_qrx2_x_absolute(:);
vehicle.ego.tx3_qrx2.y       = ego_tx3_qrx2_y_absolute(:);
vehicle.ego.tx4_qrx1.x       = ego_tx4_qrx1_x_absolute(:);
vehicle.ego.tx4_qrx1.y       = ego_tx4_qrx1_y_absolute(:);

% Save output for the relatives (to vehicle.ego.tx4_qrx1)
vehicle.target_relative.front_center.x = vehicle.target.front_center.x - vehicle.ego.tx4_qrx1.x;
vehicle.target_relative.front_center.y = vehicle.target.front_center.y - vehicle.ego.tx4_qrx1.y;
vehicle.target_relative.rear_center.x  = vehicle.target.rear_center.x - vehicle.ego.tx4_qrx1.x;
vehicle.target_relative.rear_center.y  = vehicle.target.rear_center.y - vehicle.ego.tx4_qrx1.y;
vehicle.target_relative.tx1_qrx4.x     = vehicle.target.tx1_qrx4.x - vehicle.ego.tx4_qrx1.x;
vehicle.target_relative.tx1_qrx4.y     = vehicle.target.tx1_qrx4.y - vehicle.ego.tx4_qrx1.y;
vehicle.target_relative.tx2_qrx3.x     = vehicle.target.tx2_qrx3.x - vehicle.ego.tx4_qrx1.x;
vehicle.target_relative.tx2_qrx3.y     = vehicle.target.tx2_qrx3.y - vehicle.ego.tx4_qrx1.y;
vehicle.target_relative.tx3_qrx2.x     = vehicle.target.tx3_qrx2.x - vehicle.ego.tx4_qrx1.x;
vehicle.target_relative.tx3_qrx2.y     = vehicle.target.tx3_qrx2.y - vehicle.ego.tx4_qrx1.y;
vehicle.target_relative.tx4_qrx1.x     = vehicle.target.tx4_qrx1.x - vehicle.ego.tx4_qrx1.x;
vehicle.target_relative.tx4_qrx1.y     = vehicle.target.tx4_qrx1.y - vehicle.ego.tx4_qrx1.y;
vehicle.target_relative.heading        = vehicle.target.heading - vehicle.ego.heading;

% We're not interested in the individual TX unit speeds or rear center 
% speed, so dropping those. Dropping this actually fits with the
% assumptions of AoA1, which assume that heading doesn't change during VLP
% cycle -> if heading doesn't change, all points on the vehicle have the
% same speed since it's rigid.

% vehCfg_video(vehicle);
% answer = questdlg('The video looks OK?', 'Continue?', 'Yes','No (stops script)','No (stops script)');
% % Handle response
% switch answer
%     case 'Yes'
%     case 'No (stops script)'
%         clear all;
%         return;
% end

answer = inputdlg('Enter filename for the vehicular configuration file','Vehicular Configuration Filename',[1 50],{'vehCfg_<explanation>.mat'});
save(strcat('data/',answer{1}),'vehicle')

