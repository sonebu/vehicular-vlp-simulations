clear all;  close all;  clc;
addpath('fcn');

%% Load VLC Configuration 
[fn, pt] = uigetfile('../00_vlcCfg/data/*.mat','Load VLC Configuration');
load(strcat(pt,fn));

%% Load Vehicular Configuration
[fn, pt] = uigetfile('../01_vehCfg/data/*.mat','Load Vehicular Configuration');
load(strcat(pt,fn));

clearvars -except qrx tx vehicle

%% Simulate 
% target leading, ego following, so tx1/2 to qrx1/2
vehTime = vehicle.t.values;
tx2_to_qrx1_gain = zeros(length(vehTime),1);
tx2_to_qrx2_gain = zeros(length(vehTime),1);
tx1_to_qrx1_gain = zeros(length(vehTime),1);
tx1_to_qrx2_gain = zeros(length(vehTime),1);
tx2_to_qrx1_delay = zeros(length(vehTime),1);
tx2_to_qrx2_delay = zeros(length(vehTime),1);
tx1_to_qrx1_delay = zeros(length(vehTime),1);
tx1_to_qrx2_delay = zeros(length(vehTime),1);

f = waitbar(0, 'Generating data for this trajectory');
set(findall(f),'Units', 'normalized');
set(f,'Position', [0.25 0.4 0.5 0.1]);
A_share_tx2_to_qrx1 = zeros(length(vehTime),1); B_share_tx2_to_qrx1 = zeros(length(vehTime),1); C_share_tx2_to_qrx1 = zeros(length(vehTime),1); D_share_tx2_to_qrx1 = zeros(length(vehTime),1);
A_share_tx2_to_qrx2 = zeros(length(vehTime),1); B_share_tx2_to_qrx2 = zeros(length(vehTime),1); C_share_tx2_to_qrx2 = zeros(length(vehTime),1); D_share_tx2_to_qrx2 = zeros(length(vehTime),1);
A_share_tx1_to_qrx1 = zeros(length(vehTime),1); B_share_tx1_to_qrx1 = zeros(length(vehTime),1); C_share_tx1_to_qrx1 = zeros(length(vehTime),1); D_share_tx1_to_qrx1 = zeros(length(vehTime),1);
A_share_tx1_to_qrx2 = zeros(length(vehTime),1); B_share_tx1_to_qrx2 = zeros(length(vehTime),1); C_share_tx1_to_qrx2 = zeros(length(vehTime),1); D_share_tx1_to_qrx2 = zeros(length(vehTime),1);

tt             = vehicle.target_relative;
speed_of_light = 299792458; % [m/s]

a = 0;
avg_a = 0;
for i=1:length(vehTime)
    total_estimated_minutes         = floor(avg_a*length(vehTime)/60);
    total_estimated_seconds         = round(avg_a*length(vehTime) - total_estimated_minutes*60);
    remaining_estimated_minutes     = floor(avg_a*(length(vehTime)-i)/60);
    remaining_estimated_seconds     = round(avg_a*(length(vehTime)-i) - remaining_estimated_minutes*60);
    tic
    waitbar(i/length(vehTime),f, sprintf('Generating data for this trajectory, estimated remaining time: %02d:%02d/%02d:%02d', remaining_estimated_minutes, remaining_estimated_seconds, total_estimated_minutes, total_estimated_seconds));

    % Compute qrx gains (optical!)
    tx2_to_qrx1_gain(i)  = v2lcDataGen_rxGain(tx.pattern.map, tt.tx2_qrx3.x(i)                    , tt.tx2_qrx3.y(i), 0, qrx.f_QRX.params.d_L);
    tx2_to_qrx2_gain(i)  = v2lcDataGen_rxGain(tx.pattern.map, tt.tx2_qrx3.x(i) - vehicle.ego.width, tt.tx2_qrx3.y(i), 0, qrx.f_QRX.params.d_L);
    tx1_to_qrx1_gain(i)  = v2lcDataGen_rxGain(tx.pattern.map, tt.tx1_qrx4.x(i)                    , tt.tx1_qrx4.y(i), 0, qrx.f_QRX.params.d_L);
    tx1_to_qrx2_gain(i)  = v2lcDataGen_rxGain(tx.pattern.map, tt.tx1_qrx4.x(i) - vehicle.ego.width, tt.tx1_qrx4.y(i), 0, qrx.f_QRX.params.d_L);

    tx2_to_qrx1_delay(i) = sqrt(tt.tx2_qrx3.x(i).^2 + tt.tx2_qrx3.y(i).^2)/speed_of_light;
    tx2_to_qrx2_delay(i) = sqrt((tt.tx2_qrx3.x(i) - vehicle.ego.width).^2 + tt.tx2_qrx3.y(i).^2)/speed_of_light;
    tx1_to_qrx1_delay(i) = sqrt(tt.tx1_qrx4.x(i).^2 + tt.tx1_qrx4.y(i).^2)/speed_of_light;
    tx1_to_qrx2_delay(i) = sqrt((tt.tx1_qrx4.x(i) - vehicle.ego.width).^2 + tt.tx1_qrx4.y(i).^2)/speed_of_light;
    
    % Compute shares for each qrx cell, on each qrx, for each TX, so 4 runs in total
 	[ A_share_tx2_to_qrx1(i), B_share_tx2_to_qrx1(i), C_share_tx2_to_qrx1(i), D_share_tx2_to_qrx1(i) ] = v2lcDataGen_qdMeas(0, tt.tx2_qrx3.y(i), tt.tx2_qrx3.x(i), qrx);
 	[ A_share_tx2_to_qrx2(i), B_share_tx2_to_qrx2(i), C_share_tx2_to_qrx2(i), D_share_tx2_to_qrx2(i) ] = v2lcDataGen_qdMeas(0, tt.tx2_qrx3.y(i), tt.tx2_qrx3.x(i) - vehicle.ego.width, qrx);
 	[ A_share_tx1_to_qrx1(i), B_share_tx1_to_qrx1(i), C_share_tx1_to_qrx1(i), D_share_tx1_to_qrx1(i) ] = v2lcDataGen_qdMeas(0, tt.tx1_qrx4.y(i), tt.tx1_qrx4.x(i), qrx);
 	[ A_share_tx1_to_qrx2(i), B_share_tx1_to_qrx2(i), C_share_tx1_to_qrx2(i), D_share_tx1_to_qrx2(i) ] = v2lcDataGen_qdMeas(0, tt.tx1_qrx4.y(i), tt.tx1_qrx4.x(i) - vehicle.ego.width, qrx);
    a = toc;
    avg_a = (avg_a*(i-1) + a)/i;
end
% disp(['took ' num2str(a/60) ' minutes'])
clear a total_estimated_minutes total_estimated_seconds remaining_estimated_minutes remaining_estimated_seconds i; close(f);
tx2_to_qrx1A_watts = tx2_to_qrx1_gain.*A_share_tx2_to_qrx1.*tx.power; tx2_to_qrx1B_watts = tx2_to_qrx1_gain.*B_share_tx2_to_qrx1.*tx.power;
tx2_to_qrx1C_watts = tx2_to_qrx1_gain.*C_share_tx2_to_qrx1.*tx.power; tx2_to_qrx1D_watts = tx2_to_qrx1_gain.*D_share_tx2_to_qrx1.*tx.power;
tx2_to_qrx2A_watts = tx2_to_qrx2_gain.*A_share_tx2_to_qrx2.*tx.power; tx2_to_qrx2B_watts = tx2_to_qrx2_gain.*B_share_tx2_to_qrx2.*tx.power;
tx2_to_qrx2C_watts = tx2_to_qrx2_gain.*C_share_tx2_to_qrx2.*tx.power; tx2_to_qrx2D_watts = tx2_to_qrx2_gain.*D_share_tx2_to_qrx2.*tx.power;
tx1_to_qrx1A_watts = tx1_to_qrx1_gain.*A_share_tx1_to_qrx1.*tx.power; tx1_to_qrx1B_watts = tx1_to_qrx1_gain.*B_share_tx1_to_qrx1.*tx.power;
tx1_to_qrx1C_watts = tx1_to_qrx1_gain.*C_share_tx1_to_qrx1.*tx.power; tx1_to_qrx1D_watts = tx1_to_qrx1_gain.*D_share_tx1_to_qrx1.*tx.power;
tx1_to_qrx2A_watts = tx1_to_qrx2_gain.*A_share_tx1_to_qrx2.*tx.power; tx1_to_qrx2B_watts = tx1_to_qrx2_gain.*B_share_tx1_to_qrx2.*tx.power;
tx1_to_qrx2C_watts = tx1_to_qrx2_gain.*C_share_tx1_to_qrx2.*tx.power; tx1_to_qrx2D_watts = tx1_to_qrx2_gain.*D_share_tx1_to_qrx2.*tx.power;

figure
plot(vehTime,tx2_to_qrx1A_watts), hold on
plot(vehTime,tx2_to_qrx1B_watts,'r'), hold on
plot(vehTime,tx2_to_qrx1C_watts,'g'), hold on
plot(vehTime,tx2_to_qrx1D_watts,'c'), hold off

figure
plot(vehTime,tx2_to_qrx2A_watts), hold on
plot(vehTime,tx2_to_qrx2B_watts,'r'), hold on
plot(vehTime,tx2_to_qrx2C_watts,'g'), hold on
plot(vehTime,tx2_to_qrx2D_watts,'c'), hold off

% Pack for algorithm simulations
channel.qrx1.power.tx2.A = tx2_to_qrx1A_watts; channel.qrx1.power.tx2.B = tx2_to_qrx1B_watts;
channel.qrx1.power.tx2.C = tx2_to_qrx1C_watts; channel.qrx1.power.tx2.D = tx2_to_qrx1D_watts;
channel.qrx2.power.tx2.A = tx2_to_qrx2A_watts; channel.qrx2.power.tx2.B = tx2_to_qrx2B_watts;
channel.qrx2.power.tx2.C = tx2_to_qrx2C_watts; channel.qrx2.power.tx2.D = tx2_to_qrx2D_watts;
channel.qrx1.power.tx1.A = tx1_to_qrx1A_watts; channel.qrx1.power.tx1.B = tx1_to_qrx1B_watts;
channel.qrx1.power.tx1.C = tx1_to_qrx1C_watts; channel.qrx1.power.tx1.D = tx1_to_qrx1D_watts;
channel.qrx2.power.tx1.A = tx1_to_qrx2A_watts; channel.qrx2.power.tx1.B = tx1_to_qrx2B_watts;
channel.qrx2.power.tx1.C = tx1_to_qrx2C_watts; channel.qrx2.power.tx1.D = tx1_to_qrx2D_watts;

channel.qrx1.delay.tx2 = tx2_to_qrx1_delay;
channel.qrx1.delay.tx1 = tx1_to_qrx1_delay;
channel.qrx2.delay.tx2 = tx2_to_qrx2_delay;
channel.qrx2.delay.tx1 = tx1_to_qrx2_delay;

answer = inputdlg('Enter filename for the simulation output file','Simulation Output Filename',[1 50],{'v2lcRun_<explanation>.mat'});
save(strcat('data/',answer{1}),'vehicle','qrx','tx','channel');
