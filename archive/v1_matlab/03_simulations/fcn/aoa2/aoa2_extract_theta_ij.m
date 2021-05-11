function [ theta_11, theta_12, theta_21, theta_22 ] = aoa2_extract_theta_ij(vlp_dt, vlp_time, aoa2_signals, channel, qrx, soner2_params, thd, add_noise)
%aoa2_extract_theta_ij Summary of this function goes here
%   Detailed explanation goes here

sampling_period = aoa2_signals.dt;
signal_time     = [vlp_time(1):sampling_period:vlp_time(end)+vlp_dt-sampling_period];
interp_method = 'pchip';
tmp = linspace(1,length(channel.qrx1.delay.tx1),length(signal_time));
rx1_tx1_delay = interp1(1:length(channel.qrx1.delay.tx1),channel.qrx1.delay.tx1,tmp,interp_method);
qrx1_tx1_waveform    = (sin(2.*pi.*soner2_params.tx1_freq.*(signal_time-rx1_tx1_delay)))./2;
clear rx1_tx1_delay 
rx1_tx2_delay = interp1(1:length(channel.qrx1.delay.tx2),channel.qrx1.delay.tx2,tmp,interp_method);
qrx1_tx2_waveform    = (sin(2.*pi.*soner2_params.tx2_freq.*(signal_time-rx1_tx2_delay)))./2;
clear rx1_tx2_delay
rx2_tx1_delay = interp1(1:length(channel.qrx2.delay.tx1),channel.qrx2.delay.tx1,tmp,interp_method);
qrx2_tx1_waveform    = (sin(2.*pi.*soner2_params.tx1_freq.*(signal_time-rx2_tx1_delay)))./2;
clear rx2_tx1_delay
rx2_tx2_delay = interp1(1:length(channel.qrx2.delay.tx2),channel.qrx2.delay.tx2,tmp,interp_method);
qrx2_tx2_waveform    = (sin(2.*pi.*soner2_params.tx2_freq.*(signal_time-rx2_tx2_delay)))./2;
clear rx2_tx2_delay

numSamples = floor(vlp_dt/aoa2_signals.dt);
theta_11 = zeros(length(vlp_time),2); 
theta_12 = zeros(length(vlp_time),2); 
theta_21 = zeros(length(vlp_time),2); 
theta_22 = zeros(length(vlp_time),2); 
for i=1:length(vlp_time)
	qrx1A_tx1_volts = (aoa2_signals.qrx1.tx1_amps_noiseless.A( ((i-1)*numSamples+1):(i*numSamples) ) + add_noise*aoa2_signals.qrx1.noise_stdev.A( ((i-1)*numSamples+1):(i*numSamples) ).*randn(1,numSamples))*qrx.tia.params.R_F;
	qrx1B_tx1_volts = (aoa2_signals.qrx1.tx1_amps_noiseless.B( ((i-1)*numSamples+1):(i*numSamples) ) + add_noise*aoa2_signals.qrx1.noise_stdev.B( ((i-1)*numSamples+1):(i*numSamples) ).*randn(1,numSamples))*qrx.tia.params.R_F;
	qrx1C_tx1_volts = (aoa2_signals.qrx1.tx1_amps_noiseless.C( ((i-1)*numSamples+1):(i*numSamples) ) + add_noise*aoa2_signals.qrx1.noise_stdev.C( ((i-1)*numSamples+1):(i*numSamples) ).*randn(1,numSamples))*qrx.tia.params.R_F;
	qrx1D_tx1_volts = (aoa2_signals.qrx1.tx1_amps_noiseless.D( ((i-1)*numSamples+1):(i*numSamples) ) + add_noise*aoa2_signals.qrx1.noise_stdev.D( ((i-1)*numSamples+1):(i*numSamples) ).*randn(1,numSamples))*qrx.tia.params.R_F;
	qrx1A_tx2_volts = (aoa2_signals.qrx1.tx2_amps_noiseless.A( ((i-1)*numSamples+1):(i*numSamples) ) + add_noise*aoa2_signals.qrx1.noise_stdev.A( ((i-1)*numSamples+1):(i*numSamples) ).*randn(1,numSamples))*qrx.tia.params.R_F;
	qrx1B_tx2_volts = (aoa2_signals.qrx1.tx2_amps_noiseless.B( ((i-1)*numSamples+1):(i*numSamples) ) + add_noise*aoa2_signals.qrx1.noise_stdev.B( ((i-1)*numSamples+1):(i*numSamples) ).*randn(1,numSamples))*qrx.tia.params.R_F;
	qrx1C_tx2_volts = (aoa2_signals.qrx1.tx2_amps_noiseless.C( ((i-1)*numSamples+1):(i*numSamples) ) + add_noise*aoa2_signals.qrx1.noise_stdev.C( ((i-1)*numSamples+1):(i*numSamples) ).*randn(1,numSamples))*qrx.tia.params.R_F;
	qrx1D_tx2_volts = (aoa2_signals.qrx1.tx2_amps_noiseless.D( ((i-1)*numSamples+1):(i*numSamples) ) + add_noise*aoa2_signals.qrx1.noise_stdev.D( ((i-1)*numSamples+1):(i*numSamples) ).*randn(1,numSamples))*qrx.tia.params.R_F;
	qrx2A_tx1_volts = (aoa2_signals.qrx2.tx1_amps_noiseless.A( ((i-1)*numSamples+1):(i*numSamples) ) + add_noise*aoa2_signals.qrx2.noise_stdev.A( ((i-1)*numSamples+1):(i*numSamples) ).*randn(1,numSamples))*qrx.tia.params.R_F;
	qrx2B_tx1_volts = (aoa2_signals.qrx2.tx1_amps_noiseless.B( ((i-1)*numSamples+1):(i*numSamples) ) + add_noise*aoa2_signals.qrx2.noise_stdev.B( ((i-1)*numSamples+1):(i*numSamples) ).*randn(1,numSamples))*qrx.tia.params.R_F;
	qrx2C_tx1_volts = (aoa2_signals.qrx2.tx1_amps_noiseless.C( ((i-1)*numSamples+1):(i*numSamples) ) + add_noise*aoa2_signals.qrx2.noise_stdev.C( ((i-1)*numSamples+1):(i*numSamples) ).*randn(1,numSamples))*qrx.tia.params.R_F;
	qrx2D_tx1_volts = (aoa2_signals.qrx2.tx1_amps_noiseless.D( ((i-1)*numSamples+1):(i*numSamples) ) + add_noise*aoa2_signals.qrx2.noise_stdev.D( ((i-1)*numSamples+1):(i*numSamples) ).*randn(1,numSamples))*qrx.tia.params.R_F;
	qrx2A_tx2_volts = (aoa2_signals.qrx2.tx2_amps_noiseless.A( ((i-1)*numSamples+1):(i*numSamples) ) + add_noise*aoa2_signals.qrx2.noise_stdev.A( ((i-1)*numSamples+1):(i*numSamples) ).*randn(1,numSamples))*qrx.tia.params.R_F;
	qrx2B_tx2_volts = (aoa2_signals.qrx2.tx2_amps_noiseless.B( ((i-1)*numSamples+1):(i*numSamples) ) + add_noise*aoa2_signals.qrx2.noise_stdev.B( ((i-1)*numSamples+1):(i*numSamples) ).*randn(1,numSamples))*qrx.tia.params.R_F;
	qrx2C_tx2_volts = (aoa2_signals.qrx2.tx2_amps_noiseless.C( ((i-1)*numSamples+1):(i*numSamples) ) + add_noise*aoa2_signals.qrx2.noise_stdev.C( ((i-1)*numSamples+1):(i*numSamples) ).*randn(1,numSamples))*qrx.tia.params.R_F;
	qrx2D_tx2_volts = (aoa2_signals.qrx2.tx2_amps_noiseless.D( ((i-1)*numSamples+1):(i*numSamples) ) + add_noise*aoa2_signals.qrx2.noise_stdev.D( ((i-1)*numSamples+1):(i*numSamples) ).*randn(1,numSamples))*qrx.tia.params.R_F;
    
    
    [theta_11(i,1), theta_11(i,2)] = aoa_extract_theta(qrx, qrx1A_tx1_volts, qrx1B_tx1_volts, qrx1C_tx1_volts, qrx1D_tx1_volts, qrx1_tx1_waveform( ((i-1)*numSamples+1):(i*numSamples) ), thd);
    [theta_12(i,1), theta_12(i,2)] = aoa_extract_theta(qrx, qrx1A_tx2_volts, qrx1B_tx2_volts, qrx1C_tx2_volts, qrx1D_tx2_volts, qrx1_tx2_waveform( ((i-1)*numSamples+1):(i*numSamples) ), thd);
    [theta_21(i,1), theta_21(i,2)] = aoa_extract_theta(qrx, qrx2A_tx1_volts, qrx2B_tx1_volts, qrx2C_tx1_volts, qrx2D_tx1_volts, qrx2_tx1_waveform( ((i-1)*numSamples+1):(i*numSamples) ), thd);
    [theta_22(i,1), theta_22(i,2)] = aoa_extract_theta(qrx, qrx2A_tx2_volts, qrx2B_tx2_volts, qrx2C_tx2_volts, qrx2D_tx2_volts, qrx2_tx2_waveform( ((i-1)*numSamples+1):(i*numSamples) ), thd);
end

end

