function [ lut_map, lut_map_inv, fov_lim_neg, fov_lim_pos ] = vlcCfgTool_qrxMap( optical )
%VLCCFGTOOL_FOVLUT Summary of this function goes here
%   Detailed explanation goes here

%%% Plot fQRX/gQRX, Ray optics simulations basically..

d_L = optical.d_L;
d_X = optical.d_X;
d_F = optical.d_F;
d_H = optical.d_H;
d_r = (d_L/2)*(d_F-d_X)/d_F;

theta = deg2rad(-90:optical.simRes_theta:90);
y_est_arr = zeros(size(theta'));
y_est_arr1 = zeros(size(theta'));
y_arr = zeros(size(theta));

a = 0;
avg_a = 0;

f = waitbar(0, 'Running QRX FoV calculation');
set(findall(f),'Units', 'normalized');
set(f,'Position', [0.25 0.4 0.5 0.1]);
for ii = 1:length(theta)
    total_estimated_minutes         = floor(avg_a*length(theta)/60);
    total_estimated_seconds         = round(avg_a*length(theta) - total_estimated_minutes*60);
    remaining_estimated_minutes     = floor(avg_a*(length(theta)-ii)/60);
    remaining_estimated_seconds     = round(avg_a*(length(theta)-ii) - remaining_estimated_minutes*60);
    tic
    waitbar(ii/length(theta),f, sprintf('Running QRX FoV calculation, estimated remaining time: %02d:%02d/%02d:%02d', remaining_estimated_minutes, remaining_estimated_seconds, total_estimated_minutes, total_estimated_seconds));
    
    qpd_y = tan(theta(ii))*d_X;
    qpd_vertical = 0; % Future proofing for 3D
    y_arr(ii) = qpd_y;
    
    % Generate bounds for looping
    srch_x_neg = qpd_vertical-d_r;
    srch_x_pos = qpd_vertical+d_r;
    x_ar = srch_x_neg:optical.simRes_d:srch_x_pos;
    y_ar_pos = sqrt(abs(d_r.^2-(x_ar-qpd_vertical).^2))+qpd_y;
    y_ar_neg = -sqrt(abs(d_r.^2-(x_ar-qpd_vertical).^2))+qpd_y;
    
    x_filt = (x_ar <= d_H) & (x_ar >= -d_H);
    x_vals=x_ar(x_filt);
    
    y_p_vals = y_ar_pos(x_filt);
    y_p_vals(y_p_vals>d_H) = d_H;
    y_p_vals(y_p_vals<-d_H) = -d_H;
    y_n_vals = y_ar_neg(x_filt);
    y_n_vals(y_n_vals>d_H) = d_H;
    y_n_vals(y_n_vals<-d_H) = -d_H;
    
    A_ctr = 0; B_ctr = 0; C_ctr = 0; D_ctr = 0;
    for i=1:length(x_vals)
        y_vals = y_n_vals(i):optical.simRes_d:y_p_vals(i);
        % Works for uniform only! But for the spot size we use, uniform is
        % a good assumption, see Manojlovic's QPD sensitivity paper for
        % more information.
        if(x_vals(i) > 0)
            s1 = y_vals > 0;
            a = sum(s1);
            B_ctr = B_ctr + a;
            D_ctr = D_ctr + length(s1) - a;
        else
            s1 = y_vals > 0;
            a = sum(s1);
            A_ctr = A_ctr + a;
            C_ctr = C_ctr + length(s1) - a;
        end
        
    end
    
    y_est = ((A_ctr+B_ctr)-(C_ctr+D_ctr))/(A_ctr+B_ctr+C_ctr+D_ctr);
    y_est1 = ((A_ctr+B_ctr)-(C_ctr+D_ctr))/(pi*(d_r^2)/(optical.simRes_d^2));
    y_est_arr(ii,1) = y_est;
    y_est_arr1(ii,1) = y_est1;
    
    a = toc;
    avg_a = (avg_a*(ii-1) + a)/ii;
end
close(f)

figure
theta_arr = -90:optical.simRes_theta:90;
plot(theta_arr, y_est_arr,'g','LineWidth', 1.5), grid on, hold on
oob = find(abs(y_est_arr) == 1)-length(y_est_arr)/2;
fov_lim_neg = theta_arr(length(y_est_arr)/2-min(abs(oob)));
fov_lim_pos = theta_arr(length(y_est_arr)/2+min(abs(oob))+1);
plot([fov_lim_neg fov_lim_neg], [-2 2],'c','LineWidth',1)
plot([fov_lim_pos fov_lim_pos], [-2 2],'c','LineWidth',1)
plot([fov_lim_neg fov_lim_pos], [-1 1],'k--','LineWidth',1)
% lut = [theta_arr(length(y_est_arr)/2-min(abs(oob)) : length(y_est_arr)/2+min(abs(oob))+1);
%        y_est_arr(length(y_est_arr)/2-min(abs(oob)) : length(y_est_arr)/2+min(abs(oob))+1)];
lut = [theta_arr;y_est_arr1'];
xlim([-90 90]);
ylim([-1 1]);
xlabel('Theta (deg)')
ylabel('Spot Position (p.u.)')
title('g_{QRX} Mapping')
hold off

lut_crop(1,:) = lut(1,find(lut(1,:)==fov_lim_neg):find(lut(1,:)==fov_lim_pos));
lut_crop(2,:) = lut(2,find(lut(1,:)==fov_lim_neg):find(lut(1,:)==fov_lim_pos));

% lut_map     = fit(lut_crop(1,:)', lut_crop(2,:)', 'smoothingspline','SmoothingParam',0.2);
% lut_map_inv = fit(lut_crop(2,:)', lut_crop(1,:)', 'smoothingspline','SmoothingParam',0.2);

lut_map     = fit(lut_crop(1,:)', lut_crop(2,:)', 'cubicinterp');
lut_map_inv = fit(lut_crop(2,:)', lut_crop(1,:)', 'cubicinterp');

end

