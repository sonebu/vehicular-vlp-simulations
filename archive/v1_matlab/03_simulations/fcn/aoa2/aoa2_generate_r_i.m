function [ signals, snrs ] = aoa2_generate_r_i(qrx, channel, vehicle, sunlight_scaler, circuit_temperature, soner2_params)
%aoa2_generate_r_i Summary of this function goes here
%   Detailed explanation goes here

% Noise generation
I_bg = (min(1, max(0, sunlight_scaler))*5100)*1e-6; % ref: A.J.C. Moreira, "Optical interference produced by artificial light"
qrx1A_opticalPower = channel.qrx1.power.tx1.A + channel.qrx1.power.tx2.A;
qrx1B_opticalPower = channel.qrx1.power.tx1.B + channel.qrx1.power.tx2.B;
qrx1C_opticalPower = channel.qrx1.power.tx1.C + channel.qrx1.power.tx2.C;
qrx1D_opticalPower = channel.qrx1.power.tx1.D + channel.qrx1.power.tx2.D;
qrx2A_opticalPower = channel.qrx2.power.tx1.A + channel.qrx2.power.tx2.A;
qrx2B_opticalPower = channel.qrx2.power.tx1.B + channel.qrx2.power.tx2.B;
qrx2C_opticalPower = channel.qrx2.power.tx1.C + channel.qrx2.power.tx2.C;
qrx2D_opticalPower = channel.qrx2.power.tx1.D + channel.qrx2.power.tx2.D;

% /16 due to C_T^2 in the thermal_factor2, each cell gets 1/4 of the total cap
qrx1A_noise_var_amps2 =   qrx.tia.shot_P_r_factor*qrx1A_opticalPower + qrx.tia.shot_I_bg_factor*I_bg ...
                        + circuit_temperature*(qrx.tia.thermal_factor1 + qrx.tia.thermal_factor2/16); 
qrx1A_noise_std_amps  = sqrt(qrx1A_noise_var_amps2);
qrx1B_noise_var_amps2 =   qrx.tia.shot_P_r_factor*qrx1B_opticalPower + qrx.tia.shot_I_bg_factor*I_bg ...
                        + circuit_temperature*(qrx.tia.thermal_factor1 + qrx.tia.thermal_factor2/16); 
qrx1B_noise_std_amps  = sqrt(qrx1B_noise_var_amps2);
qrx1C_noise_var_amps2 =   qrx.tia.shot_P_r_factor*qrx1C_opticalPower + qrx.tia.shot_I_bg_factor*I_bg ...
                        + circuit_temperature*(qrx.tia.thermal_factor1 + qrx.tia.thermal_factor2/16); 
qrx1C_noise_std_amps  = sqrt(qrx1C_noise_var_amps2);
qrx1D_noise_var_amps2 =   qrx.tia.shot_P_r_factor*qrx1D_opticalPower + qrx.tia.shot_I_bg_factor*I_bg ...
                        + circuit_temperature*(qrx.tia.thermal_factor1 + qrx.tia.thermal_factor2/16); 
qrx1D_noise_std_amps  = sqrt(qrx1D_noise_var_amps2);
qrx2A_noise_var_amps2 =   qrx.tia.shot_P_r_factor*qrx2A_opticalPower + qrx.tia.shot_I_bg_factor*I_bg ...
                        + circuit_temperature*(qrx.tia.thermal_factor1 + qrx.tia.thermal_factor2/16); 
qrx2A_noise_std_amps  = sqrt(qrx2A_noise_var_amps2);
qrx2B_noise_var_amps2 =   qrx.tia.shot_P_r_factor*qrx2B_opticalPower + qrx.tia.shot_I_bg_factor*I_bg ...
                        + circuit_temperature*(qrx.tia.thermal_factor1 + qrx.tia.thermal_factor2/16); 
qrx2B_noise_std_amps  = sqrt(qrx2B_noise_var_amps2);
qrx2C_noise_var_amps2 =   qrx.tia.shot_P_r_factor*qrx2C_opticalPower + qrx.tia.shot_I_bg_factor*I_bg ...
                        + circuit_temperature*(qrx.tia.thermal_factor1 + qrx.tia.thermal_factor2/16); 
qrx2C_noise_std_amps  = sqrt(qrx2C_noise_var_amps2);
qrx2D_noise_var_amps2 =   qrx.tia.shot_P_r_factor*qrx2D_opticalPower + qrx.tia.shot_I_bg_factor*I_bg ...
                        + circuit_temperature*(qrx.tia.thermal_factor1 + qrx.tia.thermal_factor2/16); 
qrx2D_noise_std_amps  = sqrt(qrx2D_noise_var_amps2);
                 
% Signal generation
qrx1A_tx1_peakAmps = channel.qrx1.power.tx1.A*qrx.tia.params.gamma;
qrx1B_tx1_peakAmps = channel.qrx1.power.tx1.B*qrx.tia.params.gamma;
qrx1C_tx1_peakAmps = channel.qrx1.power.tx1.C*qrx.tia.params.gamma;
qrx1D_tx1_peakAmps = channel.qrx1.power.tx1.D*qrx.tia.params.gamma;
qrx1A_tx2_peakAmps = channel.qrx1.power.tx2.A*qrx.tia.params.gamma;
qrx1B_tx2_peakAmps = channel.qrx1.power.tx2.B*qrx.tia.params.gamma;
qrx1C_tx2_peakAmps = channel.qrx1.power.tx2.C*qrx.tia.params.gamma;
qrx1D_tx2_peakAmps = channel.qrx1.power.tx2.D*qrx.tia.params.gamma;
qrx2A_tx1_peakAmps = channel.qrx2.power.tx1.A*qrx.tia.params.gamma;
qrx2B_tx1_peakAmps = channel.qrx2.power.tx1.B*qrx.tia.params.gamma;
qrx2C_tx1_peakAmps = channel.qrx2.power.tx1.C*qrx.tia.params.gamma;
qrx2D_tx1_peakAmps = channel.qrx2.power.tx1.D*qrx.tia.params.gamma;
qrx2A_tx2_peakAmps = channel.qrx2.power.tx2.A*qrx.tia.params.gamma;
qrx2B_tx2_peakAmps = channel.qrx2.power.tx2.B*qrx.tia.params.gamma;
qrx2C_tx2_peakAmps = channel.qrx2.power.tx2.C*qrx.tia.params.gamma;
qrx2D_tx2_peakAmps = channel.qrx2.power.tx2.D*qrx.tia.params.gamma;

% rng(1);
sampling_period = soner2_params.dt;
signal_time = [vehicle.t.start:sampling_period:vehicle.t.stop-sampling_period];
tx1_wavFreq = soner2_params.tx1_freq; % Hz
tx2_wavFreq = soner2_params.tx2_freq; % Hz

interp_method = 'pchip';
tmp = linspace(1,length(qrx1A_tx1_peakAmps),length(signal_time));
rx1_tx1_delay = interp1(1:length(channel.qrx1.delay.tx1),channel.qrx1.delay.tx1,tmp,interp_method);
rx1_tx2_delay = interp1(1:length(channel.qrx1.delay.tx2),channel.qrx1.delay.tx2,tmp,interp_method);
rx2_tx1_delay = interp1(1:length(channel.qrx2.delay.tx1),channel.qrx2.delay.tx1,tmp,interp_method);
rx2_tx2_delay = interp1(1:length(channel.qrx2.delay.tx2),channel.qrx2.delay.tx2,tmp,interp_method);

qrx1A_tx1_peakAmps_sigTime = interp1(1:length(qrx1A_tx1_peakAmps),qrx1A_tx1_peakAmps,tmp,interp_method);
qrx1A_tx1_wavAmps = qrx1A_tx1_peakAmps_sigTime.*(sin(2.*pi.*tx1_wavFreq.*(signal_time-rx1_tx1_delay)))./2; % + qrx1A_noise_std_amps_sigTime.*randn(length(signal_time),1)';
signals.qrx1.tx1_amps_noiseless.A = single(qrx1A_tx1_wavAmps);
clear qrx1A_tx1_wavAmps qrx1A_tx1_peakAmps_sigTime

qrx1B_tx1_peakAmps_sigTime = interp1(1:length(qrx1B_tx1_peakAmps),qrx1B_tx1_peakAmps,tmp,interp_method);
qrx1B_tx1_wavAmps = qrx1B_tx1_peakAmps_sigTime.*(sin(2.*pi.*tx1_wavFreq.*(signal_time-rx1_tx1_delay)))./2; % + qrx1B_noise_std_amps_sigTime.*randn(length(signal_time),1)';
signals.qrx1.tx1_amps_noiseless.B = single(qrx1B_tx1_wavAmps);
clear qrx1B_tx1_wavAmps qrx1B_tx1_peakAmps_sigTime

qrx1C_tx1_peakAmps_sigTime = interp1(1:length(qrx1C_tx1_peakAmps),qrx1C_tx1_peakAmps,tmp,interp_method);
qrx1C_tx1_wavAmps = qrx1C_tx1_peakAmps_sigTime.*(sin(2.*pi.*tx1_wavFreq.*(signal_time-rx1_tx1_delay)))./2; % + qrx1C_noise_std_amps_sigTime.*randn(length(signal_time),1)';
signals.qrx1.tx1_amps_noiseless.C = single(qrx1C_tx1_wavAmps);
clear qrx1C_tx1_wavAmps qrx1C_tx1_peakAmps_sigTime

qrx1D_tx1_peakAmps_sigTime = interp1(1:length(qrx1D_tx1_peakAmps),qrx1D_tx1_peakAmps,tmp,interp_method);
qrx1D_tx1_wavAmps = qrx1D_tx1_peakAmps_sigTime.*(sin(2.*pi.*tx1_wavFreq.*(signal_time-rx1_tx1_delay)))./2; % + qrx1D_noise_std_amps_sigTime.*randn(length(signal_time),1)';
signals.qrx1.tx1_amps_noiseless.D = single(qrx1D_tx1_wavAmps);
clear qrx1D_tx1_wavAmps qrx1D_tx1_peakAmps_sigTime

qrx1A_tx2_peakAmps_sigTime = interp1(1:length(qrx1A_tx2_peakAmps),qrx1A_tx2_peakAmps,tmp,interp_method);
qrx1A_tx2_wavAmps = qrx1A_tx2_peakAmps_sigTime.*(sin(2.*pi.*tx2_wavFreq.*(signal_time-rx1_tx2_delay)))./2; % + qrx1A_noise_std_amps_sigTime.*randn(length(signal_time),1)';
signals.qrx1.tx2_amps_noiseless.A = single(qrx1A_tx2_wavAmps);
clear qrx1A_tx2_wavAmps qrx1A_tx2_peakAmps_sigTime

qrx1B_tx2_peakAmps_sigTime = interp1(1:length(qrx1B_tx2_peakAmps),qrx1B_tx2_peakAmps,tmp,interp_method);
qrx1B_tx2_wavAmps = qrx1B_tx2_peakAmps_sigTime.*(sin(2.*pi.*tx2_wavFreq.*(signal_time-rx1_tx2_delay)))./2; % + qrx1B_noise_std_amps_sigTime.*randn(length(signal_time),1)';
signals.qrx1.tx2_amps_noiseless.B = single(qrx1B_tx2_wavAmps);
clear qrx1B_tx2_wavAmps qrx1B_tx2_peakAmps_sigTime

qrx1C_tx2_peakAmps_sigTime = interp1(1:length(qrx1C_tx2_peakAmps),qrx1C_tx2_peakAmps,tmp,interp_method);
qrx1C_tx2_wavAmps = qrx1C_tx2_peakAmps_sigTime.*(sin(2.*pi.*tx2_wavFreq.*(signal_time-rx1_tx2_delay)))./2; % + qrx1C_noise_std_amps_sigTime.*randn(length(signal_time),1)';
signals.qrx1.tx2_amps_noiseless.C = single(qrx1C_tx2_wavAmps);
clear qrx1C_tx2_wavAmps qrx1C_tx2_peakAmps_sigTime

qrx1D_tx2_peakAmps_sigTime = interp1(1:length(qrx1D_tx2_peakAmps),qrx1D_tx2_peakAmps,tmp,interp_method);
qrx1D_tx2_wavAmps = qrx1D_tx2_peakAmps_sigTime.*(sin(2.*pi.*tx2_wavFreq.*(signal_time-rx1_tx2_delay)))./2; % + qrx1D_noise_std_amps_sigTime.*randn(length(signal_time),1)';
signals.qrx1.tx2_amps_noiseless.D = single(qrx1D_tx2_wavAmps);
clear qrx1D_tx2_wavAmps qrx1D_tx2_peakAmps_sigTime

qrx2A_tx1_peakAmps_sigTime = interp1(1:length(qrx2A_tx1_peakAmps),qrx2A_tx1_peakAmps,tmp,interp_method);
qrx2A_tx1_wavAmps = qrx2A_tx1_peakAmps_sigTime.*(sin(2.*pi.*tx1_wavFreq.*(signal_time-rx2_tx1_delay)))./2; % + qrx2A_noise_std_amps_sigTime.*randn(length(signal_time),1)';
signals.qrx2.tx1_amps_noiseless.A = single(qrx2A_tx1_wavAmps);
clear qrx2A_tx1_wavAmps qrx2A_tx1_peakAmps_sigTime

qrx2B_tx1_peakAmps_sigTime = interp1(1:length(qrx2B_tx1_peakAmps),qrx2B_tx1_peakAmps,tmp,interp_method);
qrx2B_tx1_wavAmps = qrx2B_tx1_peakAmps_sigTime.*(sin(2.*pi.*tx1_wavFreq.*(signal_time-rx2_tx1_delay)))./2; % + qrx2B_noise_std_amps_sigTime.*randn(length(signal_time),1)';
signals.qrx2.tx1_amps_noiseless.B = single(qrx2B_tx1_wavAmps);
clear qrx2B_tx1_wavAmps qrx2B_tx1_peakAmps_sigTime

qrx2C_tx1_peakAmps_sigTime = interp1(1:length(qrx2C_tx1_peakAmps),qrx2C_tx1_peakAmps,tmp,interp_method);
qrx2C_tx1_wavAmps = qrx2C_tx1_peakAmps_sigTime.*(sin(2.*pi.*tx1_wavFreq.*(signal_time-rx2_tx1_delay)))./2; % + qrx2C_noise_std_amps_sigTime.*randn(length(signal_time),1)';
signals.qrx2.tx1_amps_noiseless.C = single(qrx2C_tx1_wavAmps);
clear qrx2C_tx1_wavAmps qrx2C_tx1_peakAmps_sigTime

qrx2D_tx1_peakAmps_sigTime = interp1(1:length(qrx2D_tx1_peakAmps),qrx2D_tx1_peakAmps,tmp,interp_method);
qrx2D_tx1_wavAmps = qrx2D_tx1_peakAmps_sigTime.*(sin(2.*pi.*tx1_wavFreq.*(signal_time-rx2_tx1_delay)))./2; % + qrx2D_noise_std_amps_sigTime.*randn(length(signal_time),1)';
signals.qrx2.tx1_amps_noiseless.D = single(qrx2D_tx1_wavAmps);
clear qrx2D_tx1_wavAmps qrx2D_tx1_peakAmps_sigTime

qrx2A_tx2_peakAmps_sigTime = interp1(1:length(qrx2A_tx2_peakAmps),qrx2A_tx2_peakAmps,tmp,interp_method);
qrx2A_tx2_wavAmps = qrx2A_tx2_peakAmps_sigTime.*(sin(2.*pi.*tx2_wavFreq.*(signal_time-rx2_tx2_delay)))./2; % + qrx2A_noise_std_amps_sigTime.*randn(length(signal_time),1)';
signals.qrx2.tx2_amps_noiseless.A = single(qrx2A_tx2_wavAmps);
clear qrx2A_tx2_wavAmps qrx2A_tx2_peakAmps_sigTime

qrx2B_tx2_peakAmps_sigTime = interp1(1:length(qrx2B_tx2_peakAmps),qrx2B_tx2_peakAmps,tmp,interp_method);
qrx2B_tx2_wavAmps = qrx2B_tx2_peakAmps_sigTime.*(sin(2.*pi.*tx2_wavFreq.*(signal_time-rx2_tx2_delay)))./2; % + qrx2B_noise_std_amps_sigTime.*randn(length(signal_time),1)';
signals.qrx2.tx2_amps_noiseless.B = single(qrx2B_tx2_wavAmps);
clear qrx2B_tx2_wavAmps qrx2B_tx2_peakAmps_sigTime

qrx2C_tx2_peakAmps_sigTime = interp1(1:length(qrx2C_tx2_peakAmps),qrx2C_tx2_peakAmps,tmp,interp_method);
qrx2C_tx2_wavAmps = qrx2C_tx2_peakAmps_sigTime.*(sin(2.*pi.*tx2_wavFreq.*(signal_time-rx2_tx2_delay)))./2; % + qrx2C_noise_std_amps_sigTime.*randn(length(signal_time),1)';
signals.qrx2.tx2_amps_noiseless.C = single(qrx2C_tx2_wavAmps);
clear qrx2C_tx2_wavAmps qrx2C_tx2_peakAmps_sigTime

qrx2D_tx2_peakAmps_sigTime = interp1(1:length(qrx2D_tx2_peakAmps),qrx2D_tx2_peakAmps,tmp,interp_method);
qrx2D_tx2_wavAmps = qrx2D_tx2_peakAmps_sigTime.*(sin(2.*pi.*tx2_wavFreq.*(signal_time-rx2_tx2_delay)))./2; % + qrx2D_noise_std_amps_sigTime.*randn(length(signal_time),1)';
signals.qrx2.tx2_amps_noiseless.D = single(qrx2D_tx2_wavAmps);
clear qrx2D_tx2_wavAmps qrx2D_tx2_peakAmps_sigTime

clear rx1_tx1_delay rx1_tx2_delay rx2_tx1_delay rx2_tx2_delay 

qrx1A_noise_std_amps_sigTime = interp1(1:length(qrx1A_noise_std_amps),qrx1A_noise_std_amps,tmp,interp_method);
signals.qrx1.noise_stdev.A = single(qrx1A_noise_std_amps_sigTime);
clear qrx1A_noise_std_amps_sigTime 

qrx1B_noise_std_amps_sigTime = interp1(1:length(qrx1B_noise_std_amps),qrx1B_noise_std_amps,tmp,interp_method);
signals.qrx1.noise_stdev.B = single(qrx1B_noise_std_amps_sigTime);
clear qrx1B_noise_std_amps_sigTime

qrx1C_noise_std_amps_sigTime = interp1(1:length(qrx1C_noise_std_amps),qrx1C_noise_std_amps,tmp,interp_method);
signals.qrx1.noise_stdev.C = single(qrx1C_noise_std_amps_sigTime);
clear qrx1C_noise_std_amps_sigTime 

qrx1D_noise_std_amps_sigTime = interp1(1:length(qrx1D_noise_std_amps),qrx1D_noise_std_amps,tmp,interp_method);
signals.qrx1.noise_stdev.D = single(qrx1D_noise_std_amps_sigTime);
clear qrx1D_noise_std_amps_sigTime

qrx2A_noise_std_amps_sigTime = interp1(1:length(qrx2A_noise_std_amps),qrx2A_noise_std_amps,tmp,interp_method);
signals.qrx2.noise_stdev.A = single(qrx2A_noise_std_amps_sigTime);
clear qrx2A_noise_std_amps_sigTime

qrx2B_noise_std_amps_sigTime = interp1(1:length(qrx2B_noise_std_amps),qrx2B_noise_std_amps,tmp,interp_method);
signals.qrx2.noise_stdev.B = single(qrx2B_noise_std_amps_sigTime);
clear qrx2B_noise_std_amps_sigTime

qrx2C_noise_std_amps_sigTime = interp1(1:length(qrx2C_noise_std_amps),qrx2C_noise_std_amps,tmp,interp_method);
signals.qrx2.noise_stdev.C = single(qrx2C_noise_std_amps_sigTime);
clear qrx2C_noise_std_amps_sigTime

qrx2D_noise_std_amps_sigTime = interp1(1:length(qrx2D_noise_std_amps),qrx2D_noise_std_amps,tmp,interp_method);
signals.qrx2.noise_stdev.D = single(qrx2D_noise_std_amps_sigTime);
clear qrx2D_noise_std_amps_sigTime

signals.time = signal_time;
clear signal_time 

signals.dt = sampling_period;

% SNR computation
qrx1A_tx1_rmsAmps = qrx1A_tx1_peakAmps/(2*sqrt(2));
qrx1B_tx1_rmsAmps = qrx1B_tx1_peakAmps/(2*sqrt(2));
qrx1C_tx1_rmsAmps = qrx1C_tx1_peakAmps/(2*sqrt(2));
qrx1D_tx1_rmsAmps = qrx1D_tx1_peakAmps/(2*sqrt(2));
qrx1A_tx2_rmsAmps = qrx1A_tx2_peakAmps/(2*sqrt(2));
qrx1B_tx2_rmsAmps = qrx1B_tx2_peakAmps/(2*sqrt(2));
qrx1C_tx2_rmsAmps = qrx1C_tx2_peakAmps/(2*sqrt(2));
qrx1D_tx2_rmsAmps = qrx1D_tx2_peakAmps/(2*sqrt(2));
qrx2A_tx1_rmsAmps = qrx2A_tx1_peakAmps/(2*sqrt(2));
qrx2B_tx1_rmsAmps = qrx2B_tx1_peakAmps/(2*sqrt(2));
qrx2C_tx1_rmsAmps = qrx2C_tx1_peakAmps/(2*sqrt(2));
qrx2D_tx1_rmsAmps = qrx2D_tx1_peakAmps/(2*sqrt(2));
qrx2A_tx2_rmsAmps = qrx2A_tx2_peakAmps/(2*sqrt(2));
qrx2B_tx2_rmsAmps = qrx2B_tx2_peakAmps/(2*sqrt(2));
qrx2C_tx2_rmsAmps = qrx2C_tx2_peakAmps/(2*sqrt(2));
qrx2D_tx2_rmsAmps = qrx2D_tx2_peakAmps/(2*sqrt(2));
snr_qrx1A_tx1 = 10*log10(qrx1A_tx1_rmsAmps.*qrx1A_tx1_rmsAmps./qrx1A_noise_var_amps2);
snr_qrx1B_tx1 = 10*log10(qrx1B_tx1_rmsAmps.*qrx1B_tx1_rmsAmps./qrx1B_noise_var_amps2);
snr_qrx1C_tx1 = 10*log10(qrx1C_tx1_rmsAmps.*qrx1C_tx1_rmsAmps./qrx1C_noise_var_amps2);
snr_qrx1D_tx1 = 10*log10(qrx1D_tx1_rmsAmps.*qrx1D_tx1_rmsAmps./qrx1D_noise_var_amps2);
snr_qrx1A_tx2 = 10*log10(qrx1A_tx2_rmsAmps.*qrx1A_tx2_rmsAmps./qrx1A_noise_var_amps2);
snr_qrx1B_tx2 = 10*log10(qrx1B_tx2_rmsAmps.*qrx1B_tx2_rmsAmps./qrx1B_noise_var_amps2);
snr_qrx1C_tx2 = 10*log10(qrx1C_tx2_rmsAmps.*qrx1C_tx2_rmsAmps./qrx1C_noise_var_amps2);
snr_qrx1D_tx2 = 10*log10(qrx1D_tx2_rmsAmps.*qrx1D_tx2_rmsAmps./qrx1D_noise_var_amps2);
snr_qrx2A_tx1 = 10*log10(qrx2A_tx1_rmsAmps.*qrx2A_tx1_rmsAmps./qrx2A_noise_var_amps2);
snr_qrx2B_tx1 = 10*log10(qrx2B_tx1_rmsAmps.*qrx2B_tx1_rmsAmps./qrx2B_noise_var_amps2);
snr_qrx2C_tx1 = 10*log10(qrx2C_tx1_rmsAmps.*qrx2C_tx1_rmsAmps./qrx2C_noise_var_amps2);
snr_qrx2D_tx1 = 10*log10(qrx2D_tx1_rmsAmps.*qrx2D_tx1_rmsAmps./qrx2D_noise_var_amps2);
snr_qrx2A_tx2 = 10*log10(qrx2A_tx2_rmsAmps.*qrx2A_tx2_rmsAmps./qrx2A_noise_var_amps2);
snr_qrx2B_tx2 = 10*log10(qrx2B_tx2_rmsAmps.*qrx2B_tx2_rmsAmps./qrx2B_noise_var_amps2);
snr_qrx2C_tx2 = 10*log10(qrx2C_tx2_rmsAmps.*qrx2C_tx2_rmsAmps./qrx2C_noise_var_amps2);
snr_qrx2D_tx2 = 10*log10(qrx2D_tx2_rmsAmps.*qrx2D_tx2_rmsAmps./qrx2D_noise_var_amps2);
snrs.qrx1A.tx1 = snr_qrx1A_tx1;
snrs.qrx1B.tx1 = snr_qrx1B_tx1;
snrs.qrx1C.tx1 = snr_qrx1C_tx1;
snrs.qrx1D.tx1 = snr_qrx1D_tx1;
snrs.qrx1A.tx2 = snr_qrx1A_tx2;
snrs.qrx1B.tx2 = snr_qrx1B_tx2;
snrs.qrx1C.tx2 = snr_qrx1C_tx2;
snrs.qrx1D.tx2 = snr_qrx1D_tx2;
snrs.qrx2A.tx1 = snr_qrx2A_tx1;
snrs.qrx2B.tx1 = snr_qrx2B_tx1;
snrs.qrx2C.tx1 = snr_qrx2C_tx1;
snrs.qrx2D.tx1 = snr_qrx2D_tx1;
snrs.qrx2A.tx2 = snr_qrx2A_tx2;
snrs.qrx2B.tx2 = snr_qrx2B_tx2;
snrs.qrx2C.tx2 = snr_qrx2C_tx2;
snrs.qrx2D.tx2 = snr_qrx2D_tx2;

end