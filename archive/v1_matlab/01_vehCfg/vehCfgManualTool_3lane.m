clear all; close all; clc
addpath('fcn')

%%% This is basically the same as the normal vehCfgManualTool.m but it's
%%% for generating static locations for the 3lane scenario.

%% Inputs

% Vehicle dimensions (hardcoded)
vehicle.ego.width        = 1.6;
vehicle.target.width     = 1.6;
vehicle.ego.length       = 5.0;
vehicle.target.length    = 5.0;

% Define full relative location+heading grid, no interpolation
target_front_center_x_relative  = -3:0.25:3;
target_front_center_y_relative  = [12.75:0.25:15] + vehicle.target.length;
target_heading_absolute         = -5:2.5:5;
target_heading_relative         = target_heading_absolute;
target_front_center_x_absolute  = target_front_center_x_relative;
target_front_center_y_absolute  = target_front_center_y_relative;

ego_front_center_x_absolute = zeros(size(target_front_center_x_relative));
ego_front_center_y_absolute = zeros(size(target_front_center_y_relative));
ego_heading_absolute        = zeros(size(target_heading_absolute));

vehicle.t.dt        = 2e-2; % this will be equal to 1 / estimation rate, one estimation per point
vehicle.t.start = 0;
vehicle.t.values=[];
counter = 0;
for ix = 1:length(target_front_center_x_relative)
    for iy = 1:length(target_front_center_y_relative)
        for ih = 1:length(target_heading_absolute)
            counter = counter + 1;
            vehicle.t.values = [vehicle.t.values vehicle.t.dt*(1+ (ih-1) + (iy-1)*length(target_heading_absolute) + (ix-1)*length(target_front_center_y_relative)*length(target_heading_absolute))];
            vehicle.target.front_center.x(counter) = target_front_center_x_absolute(ix);
            vehicle.target.front_center.y(counter) = target_front_center_y_absolute(iy);
            vehicle.target.heading(counter)        = target_heading_absolute(ih);
            vehicle.ego.front_center.x(counter)    = ego_front_center_x_absolute(ix);
            vehicle.ego.front_center.y(counter)    = ego_front_center_y_absolute(iy);
            vehicle.ego.heading(counter)           = ego_heading_absolute(ih);
        end
    end
end
vehicle.t.stop = vehicle.t.values(end);

% Save output for front center absolutes
vehicle.target.front_center.speed   = NaN; % no need
vehicle.ego.front_center.speed      = NaN; % no need
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
vehicle.target_relative.front_center.x = vehicle.target.front_center.x - vehicle.ego.tx4_qrx1.x';
vehicle.target_relative.front_center.y = vehicle.target.front_center.y - vehicle.ego.tx4_qrx1.y';
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

plot(vehicle.target.tx1_qrx4.x,vehicle.target.tx1_qrx4.y)

answer = inputdlg('Enter filename for the vehicular configuration file','Vehicular Configuration Filename',[1 50],{'vehCfg_<explanation>.mat'});
save(strcat('data/',answer{1}),'vehicle')

