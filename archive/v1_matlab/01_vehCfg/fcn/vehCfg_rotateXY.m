function [ new_x, new_y ] = vehCfg_rotateXY( x, y, heading )
%VEHCFG_ROTATE_XY Summary of this function goes here
%   Detailed explanation goes here

rot_mtx = zeros(2,2,length(x));
new_x = zeros(1,length(x));
new_y = zeros(1,length(x));
for i=1:length(x)
    rot_mtx(1,1,i) =  cosd(-heading(i));
    rot_mtx(1,2,i) = -sind(-heading(i));
    rot_mtx(2,1,i) =  sind(-heading(i));
    rot_mtx(2,2,i) =  cosd(-heading(i));
    rel = rot_mtx(:,:,i)*[x(i);y(i)];
    new_x(i) = rel(1);
    new_y(i) = rel(2);
end
end

