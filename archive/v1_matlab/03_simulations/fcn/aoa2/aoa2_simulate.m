function [ ] = aoa2_simulate(data_file, res_file, iterations, validAoAThd, add_noise, T_s, circuit_temperature, cond1, cond2, cond3)
%aoa2_simulate Summary of this function goes here
%   Detailed explanation goes here

%%% Load simulation output file (the algorithm only uses QRX measurements and the QRX configuration)
load(data_file);

%%% Input Parameters
t = single([0:T_s:(vehicle.t.stop)-T_s]); % seconds

if(strcmp(cond1,'night'))
    sunlight_scaler = 0.01;
elseif(strcmp(cond1,'day'))
    sunlight_scaler = 0.145;
else
    sunlight_scaler = 100;
end

% Hz
if(strcmp(cond2,'50Hz'))
    rate = 50;
elseif(strcmp(cond2,'100Hz'))
    rate = 100;
elseif(strcmp(cond2,'125Hz'))
    rate = 125;
elseif(strcmp(cond2,'200Hz'))
    rate = 200;
elseif(strcmp(cond2,'250Hz'))
    rate = 250;
else
    rate = 0;
end

% dB/m
if(strcmp(cond3,'clear'))
    atten = 0;
elseif(strcmp(cond3,'rain'))
    atten = -0.05;
elseif(strcmp(cond3,'fog'))
    atten = -0.2;
else
    atten = 0;
end
rng(1);

vlpDecimRate = round(1/(rate*vehicle.t.dt)); % algorithm gives output every *this* many cycles of movement
numTetaSmplPerPos = int32(round((1/T_s)*vehicle.t.dt)*vlpDecimRate);
vlpLen = int32(round(length(vehicle.t.values)/vlpDecimRate));
cx11 = zeros(vlpLen,iterations);
cy11 = zeros(vlpLen,iterations);
cx12 = zeros(vlpLen,iterations);
cy12 = zeros(vlpLen,iterations);
aoa_11 = zeros(vlpLen,iterations);
aoa_12 = zeros(vlpLen,iterations);
aoa_21 = zeros(vlpLen,iterations);
aoa_22 = zeros(vlpLen,iterations);

%%%
tx1_sym = 1000; % bps
tx1_wavFreqs = [5000 6000]; % Hz
tx1_wav = zeros(length(t),1);
tx1_ttlNumSyms = tx1_sym*t(end);
tx1_symNumSmpl = 1/(tx1_sym*(T_s));
for i=1:tx1_ttlNumSyms
    tx1_wav((1+(i-1)*tx1_symNumSmpl):(i*tx1_symNumSmpl)) = ...
        single(sin(2*pi*tx1_wavFreqs(1+(rand>0.5))*t((1+(i-1)*tx1_symNumSmpl):(i*tx1_symNumSmpl))));
end

tx2_sym = 1000; % bps
tx2_wavFreqs = [12000 13000]; % Hz
tx2_wav = zeros(length(t),1);
tx2_ttlNumSyms = tx2_sym*t(end);
tx2_symNumSmpl = 1/(tx2_sym*(T_s));
for i=1:tx2_ttlNumSyms
    tx2_wav((1+(i-1)*tx2_symNumSmpl):(i*tx2_symNumSmpl)) = ...
        single(sin(2*pi*tx2_wavFreqs(1+(rand>0.5))*t((1+(i-1)*tx2_symNumSmpl):(i*tx2_symNumSmpl))));
end

interp_method = 'pchip';
tmp = linspace(1,length(channel.qrx1.power.tx1.A),length(t));
d11 = sqrt(vehicle.target_relative.tx1_qrx4.x.^2 + vehicle.target_relative.tx1_qrx4.y.^2);
d12 = sqrt(vehicle.target_relative.tx2_qrx3.x.^2 + vehicle.target_relative.tx2_qrx3.y.^2);
d21 = sqrt((vehicle.target_relative.tx1_qrx4.x-vehicle.target.width).^2 + vehicle.target_relative.tx1_qrx4.y.^2);
d22 = sqrt((vehicle.target_relative.tx2_qrx3.x-vehicle.target.width).^2 + vehicle.target_relative.tx2_qrx3.y.^2);
atten11 = 10.^(d11*atten/10);
atten12 = 10.^(d12*atten/10);
atten21 = 10.^(d21*atten/10);
atten22 = 10.^(d22*atten/10);
I_bg = (min(1, max(0, sunlight_scaler))*5100)*1e-6; % ref: A.J.C. Moreira, "Optical interference produced by artificial light"

% /16 due to C_T^2 in the thermal_factor2, each cell gets 1/4 of the total cap
tx1_to_qrx1A_watts = single(interp1(1:length(channel.qrx1.power.tx1.A), atten11.*channel.qrx1.power.tx1.A ,tmp,interp_method));
tx1_to_qrx1A_sig_amps = tx1_to_qrx1A_watts'.*(tx1_wav/2).*qrx.tia.params.gamma;
tx2_to_qrx1A_watts = single(interp1(1:length(channel.qrx1.power.tx2.A), atten12.*channel.qrx1.power.tx2.A ,tmp,interp_method));
tx2_to_qrx1A_sig_amps = tx2_to_qrx1A_watts'.*(tx2_wav/2).*qrx.tia.params.gamma;
qrx1A_opticalPower = tx1_to_qrx1A_watts + tx2_to_qrx1A_watts;
qrx1A_noise_var_amps2 =   qrx.tia.shot_P_r_factor*qrx1A_opticalPower + qrx.tia.shot_I_bg_factor*I_bg ...
    + circuit_temperature*(qrx.tia.thermal_factor1 + qrx.tia.thermal_factor2/16);
qrx1A_noise_std_amps  = sqrt(qrx1A_noise_var_amps2);
clear qrx1A_opticalPower qrx1A_noise_var_amps2
clear tx1_to_qrx1A_watts tx2_to_qrx1A_watts

tx1_to_qrx1B_watts = single(interp1(1:length(channel.qrx1.power.tx1.B), atten11.*channel.qrx1.power.tx1.B ,tmp,interp_method));
tx1_to_qrx1B_sig_amps = tx1_to_qrx1B_watts'.*(tx1_wav/2).*qrx.tia.params.gamma;
tx2_to_qrx1B_watts = single(interp1(1:length(channel.qrx1.power.tx2.B), atten12.*channel.qrx1.power.tx2.B ,tmp,interp_method));
tx2_to_qrx1B_sig_amps = tx2_to_qrx1B_watts'.*(tx2_wav/2).*qrx.tia.params.gamma;
qrx1B_opticalPower = tx1_to_qrx1B_watts + tx2_to_qrx1B_watts;
qrx1B_noise_var_amps2 =   qrx.tia.shot_P_r_factor*qrx1B_opticalPower + qrx.tia.shot_I_bg_factor*I_bg ...
    + circuit_temperature*(qrx.tia.thermal_factor1 + qrx.tia.thermal_factor2/16);
qrx1B_noise_std_amps  = sqrt(qrx1B_noise_var_amps2);
clear qrx1B_opticalPower qrx1B_noise_var_amps2
clear tx1_to_qrx1B_watts tx2_to_qrx1B_watts

tx1_to_qrx1C_watts = single(interp1(1:length(channel.qrx1.power.tx1.C), atten11.*channel.qrx1.power.tx1.C ,tmp,interp_method));
tx1_to_qrx1C_sig_amps = tx1_to_qrx1C_watts'.*(tx1_wav/2).*qrx.tia.params.gamma;
tx2_to_qrx1C_watts = single(interp1(1:length(channel.qrx1.power.tx2.C), atten12.*channel.qrx1.power.tx2.C ,tmp,interp_method));
tx2_to_qrx1C_sig_amps = tx2_to_qrx1C_watts'.*(tx2_wav/2).*qrx.tia.params.gamma;
qrx1C_opticalPower = tx1_to_qrx1C_watts + tx2_to_qrx1C_watts;
qrx1C_noise_var_amps2 =   qrx.tia.shot_P_r_factor*qrx1C_opticalPower + qrx.tia.shot_I_bg_factor*I_bg ...
    + circuit_temperature*(qrx.tia.thermal_factor1 + qrx.tia.thermal_factor2/16);
qrx1C_noise_std_amps  = sqrt(qrx1C_noise_var_amps2);
clear qrx1C_opticalPower qrx1C_noise_var_amps2
clear tx1_to_qrx1C_watts tx2_to_qrx1C_watts

tx1_to_qrx1D_watts = single(interp1(1:length(channel.qrx1.power.tx1.D), atten11.*channel.qrx1.power.tx1.D ,tmp,interp_method));
tx1_to_qrx1D_sig_amps = tx1_to_qrx1D_watts'.*(tx1_wav/2).*qrx.tia.params.gamma;
tx2_to_qrx1D_watts = single(interp1(1:length(channel.qrx1.power.tx2.D), atten12.*channel.qrx1.power.tx2.D ,tmp,interp_method));
tx2_to_qrx1D_sig_amps = tx2_to_qrx1D_watts'.*(tx2_wav/2).*qrx.tia.params.gamma;
qrx1D_opticalPower = tx1_to_qrx1D_watts + tx2_to_qrx1D_watts;
qrx1D_noise_var_amps2 =   qrx.tia.shot_P_r_factor*qrx1D_opticalPower + qrx.tia.shot_I_bg_factor*I_bg ...
    + circuit_temperature*(qrx.tia.thermal_factor1 + qrx.tia.thermal_factor2/16);
qrx1D_noise_std_amps  = sqrt(qrx1D_noise_var_amps2);
clear qrx1D_opticalPower qrx1D_noise_var_amps2
clear tx1_to_qrx1D_watts tx2_to_qrx1D_watts

tx1_to_qrx2A_watts = single(interp1(1:length(channel.qrx2.power.tx1.A), atten21.*channel.qrx2.power.tx1.A ,tmp,interp_method));
tx1_to_qrx2A_sig_amps = tx1_to_qrx2A_watts'.*(tx1_wav/2).*qrx.tia.params.gamma;
tx2_to_qrx2A_watts = single(interp1(1:length(channel.qrx2.power.tx2.A), atten22.*channel.qrx2.power.tx2.A ,tmp,interp_method));
tx2_to_qrx2A_sig_amps = tx2_to_qrx2A_watts'.*(tx2_wav/2).*qrx.tia.params.gamma;
qrx2A_opticalPower = tx1_to_qrx2A_watts + tx2_to_qrx2A_watts;
qrx2A_noise_var_amps2 =   qrx.tia.shot_P_r_factor*qrx2A_opticalPower + qrx.tia.shot_I_bg_factor*I_bg ...
    + circuit_temperature*(qrx.tia.thermal_factor1 + qrx.tia.thermal_factor2/16);
qrx2A_noise_std_amps  = sqrt(qrx2A_noise_var_amps2);
clear qrx2A_opticalPower qrx2A_noise_var_amps2
clear tx1_to_qrx2A_watts tx2_to_qrx2A_watts

tx1_to_qrx2B_watts = single(interp1(1:length(channel.qrx2.power.tx1.B), atten21.*channel.qrx2.power.tx1.B ,tmp,interp_method));
tx1_to_qrx2B_sig_amps = tx1_to_qrx2B_watts'.*(tx1_wav/2).*qrx.tia.params.gamma;
tx2_to_qrx2B_watts = single(interp1(1:length(channel.qrx2.power.tx2.B), atten22.*channel.qrx2.power.tx2.B ,tmp,interp_method));
tx2_to_qrx2B_sig_amps = tx2_to_qrx2B_watts'.*(tx2_wav/2).*qrx.tia.params.gamma;
qrx2B_opticalPower = tx1_to_qrx2B_watts + tx2_to_qrx2B_watts;
qrx2B_noise_var_amps2 =   qrx.tia.shot_P_r_factor*qrx2B_opticalPower + qrx.tia.shot_I_bg_factor*I_bg ...
    + circuit_temperature*(qrx.tia.thermal_factor1 + qrx.tia.thermal_factor2/16);
qrx2B_noise_std_amps  = sqrt(qrx2B_noise_var_amps2);
clear qrx2B_opticalPower qrx2B_noise_var_amps2
clear tx1_to_qrx2B_watts tx2_to_qrx2B_watts

tx1_to_qrx2C_watts = single(interp1(1:length(channel.qrx2.power.tx1.C), atten21.*channel.qrx2.power.tx1.C ,tmp,interp_method));
tx1_to_qrx2C_sig_amps = tx1_to_qrx2C_watts'.*(tx1_wav/2).*qrx.tia.params.gamma;
tx2_to_qrx2C_watts = single(interp1(1:length(channel.qrx2.power.tx2.C), atten22.*channel.qrx2.power.tx2.C ,tmp,interp_method));
tx2_to_qrx2C_sig_amps = tx2_to_qrx2C_watts'.*(tx2_wav/2).*qrx.tia.params.gamma;
qrx2C_opticalPower = tx1_to_qrx2C_watts + tx2_to_qrx2C_watts;
qrx2C_noise_var_amps2 =   qrx.tia.shot_P_r_factor*qrx2C_opticalPower + qrx.tia.shot_I_bg_factor*I_bg ...
    + circuit_temperature*(qrx.tia.thermal_factor1 + qrx.tia.thermal_factor2/16);
qrx2C_noise_std_amps  = sqrt(qrx2C_noise_var_amps2);
clear qrx2C_opticalPower qrx2C_noise_var_amps2
clear tx1_to_qrx2C_watts tx2_to_qrx2C_watts

tx1_to_qrx2D_watts = single(interp1(1:length(channel.qrx2.power.tx1.D), atten21.*channel.qrx2.power.tx1.D ,tmp,interp_method));
tx2_to_qrx2D_watts = single(interp1(1:length(channel.qrx2.power.tx2.D), atten22.*channel.qrx2.power.tx2.D ,tmp,interp_method));
tx1_to_qrx2D_sig_amps = tx1_to_qrx2D_watts'.*(tx1_wav/2).*qrx.tia.params.gamma;
tx2_to_qrx2D_sig_amps = tx2_to_qrx2D_watts'.*(tx2_wav/2).*qrx.tia.params.gamma;
qrx2D_opticalPower = tx1_to_qrx2D_watts + tx2_to_qrx2D_watts;
qrx2D_noise_var_amps2 =   qrx.tia.shot_P_r_factor*qrx2D_opticalPower + qrx.tia.shot_I_bg_factor*I_bg ...
    + circuit_temperature*(qrx.tia.thermal_factor1 + qrx.tia.thermal_factor2/16);
qrx2D_noise_std_amps  = sqrt(qrx2D_noise_var_amps2);
clear qrx2D_opticalPower qrx2D_noise_var_amps2
clear tx1_to_qrx2D_watts tx2_to_qrx2D_watts

fii = waitbar(0, sprintf('Running aoa2 simulations, iteration: %d/%d',0,iterations));
set(findall(fii),'Units', 'normalized');
set(fii,'Position', [0.25 0.4 0.5 0.1]);
for iter=1:iterations
    waitbar(double(iter)/iterations,fii, sprintf('Running aoa2 simulations, iteration: %d/%d',iter,iterations));

    qrx1A_r_tx1 = (tx1_to_qrx1A_sig_amps + sqrt(1/2)*(add_noise*qrx1A_noise_std_amps.*randn(1,length(t)))' )*qrx.tia.params.R_F;
    qrx1A_r_tx2 = (tx2_to_qrx1A_sig_amps + sqrt(1/2)*(add_noise*qrx1A_noise_std_amps.*randn(1,length(t)))' )*qrx.tia.params.R_F;
    qrx1B_r_tx1 = (tx1_to_qrx1B_sig_amps + sqrt(1/2)*(add_noise*qrx1B_noise_std_amps.*randn(1,length(t)))' )*qrx.tia.params.R_F;
    qrx1B_r_tx2 = (tx2_to_qrx1B_sig_amps + sqrt(1/2)*(add_noise*qrx1B_noise_std_amps.*randn(1,length(t)))' )*qrx.tia.params.R_F;
    qrx1C_r_tx1 = (tx1_to_qrx1C_sig_amps + sqrt(1/2)*(add_noise*qrx1C_noise_std_amps.*randn(1,length(t)))' )*qrx.tia.params.R_F;
    qrx1C_r_tx2 = (tx2_to_qrx1C_sig_amps + sqrt(1/2)*(add_noise*qrx1C_noise_std_amps.*randn(1,length(t)))' )*qrx.tia.params.R_F;
    qrx1D_r_tx1 = (tx1_to_qrx1D_sig_amps + sqrt(1/2)*(add_noise*qrx1D_noise_std_amps.*randn(1,length(t)))' )*qrx.tia.params.R_F;
    qrx1D_r_tx2 = (tx2_to_qrx1D_sig_amps + sqrt(1/2)*(add_noise*qrx1D_noise_std_amps.*randn(1,length(t)))' )*qrx.tia.params.R_F;
    qrx2A_r_tx1 = (tx1_to_qrx2A_sig_amps + sqrt(1/2)*(add_noise*qrx2A_noise_std_amps.*randn(1,length(t)))' )*qrx.tia.params.R_F;
    qrx2A_r_tx2 = (tx2_to_qrx2A_sig_amps + sqrt(1/2)*(add_noise*qrx2A_noise_std_amps.*randn(1,length(t)))' )*qrx.tia.params.R_F;
    qrx2B_r_tx1 = (tx1_to_qrx2B_sig_amps + sqrt(1/2)*(add_noise*qrx2B_noise_std_amps.*randn(1,length(t)))' )*qrx.tia.params.R_F;
    qrx2B_r_tx2 = (tx2_to_qrx2B_sig_amps + sqrt(1/2)*(add_noise*qrx2B_noise_std_amps.*randn(1,length(t)))' )*qrx.tia.params.R_F;
    qrx2C_r_tx1 = (tx1_to_qrx2C_sig_amps + sqrt(1/2)*(add_noise*qrx2C_noise_std_amps.*randn(1,length(t)))' )*qrx.tia.params.R_F;
    qrx2C_r_tx2 = (tx2_to_qrx2C_sig_amps + sqrt(1/2)*(add_noise*qrx2C_noise_std_amps.*randn(1,length(t)))' )*qrx.tia.params.R_F;
    qrx2D_r_tx1 = (tx1_to_qrx2D_sig_amps + sqrt(1/2)*(add_noise*qrx2D_noise_std_amps.*randn(1,length(t)))' )*qrx.tia.params.R_F;
    qrx2D_r_tx2 = (tx2_to_qrx2D_sig_amps + sqrt(1/2)*(add_noise*qrx2D_noise_std_amps.*randn(1,length(t)))' )*qrx.tia.params.R_F;
    
    %%% Algorithm    
    calc_aoa_qrx1_tx1 = zeros(vlpLen,1);
    calc_aoa_qrx1_tx2 = zeros(vlpLen,1);
    calc_aoa_qrx2_tx1 = zeros(vlpLen,1);
    calc_aoa_qrx2_tx2 = zeros(vlpLen,1);
    calc_x11 = zeros(vlpLen,1);
    calc_x12 = zeros(vlpLen,1);
    calc_y11 = zeros(vlpLen,1);
    calc_y12 = zeros(vlpLen,1);
    
    for i=1:vlpLen
        % Algo: Find theta, QRX 1
        qdA = qrx1A_r_tx1(1+(i-1)*numTetaSmplPerPos:i*numTetaSmplPerPos);
        qdB = qrx1B_r_tx1(1+(i-1)*numTetaSmplPerPos:i*numTetaSmplPerPos);
        qdC = qrx1C_r_tx1(1+(i-1)*numTetaSmplPerPos:i*numTetaSmplPerPos);
        qdD = qrx1D_r_tx1(1+(i-1)*numTetaSmplPerPos:i*numTetaSmplPerPos);
        actwav =  tx1_wav(1+(i-1)*numTetaSmplPerPos:i*numTetaSmplPerPos);
        [~, calc_aoa_qrx1_tx1(i), ~, a] = vlpAlgoSim_getTeta( qrx.f_QRX.map_inv, qdA, qdB, qdC, qdD, actwav, validAoAThd);
        qdA = qrx1A_r_tx2(1+(i-1)*numTetaSmplPerPos:i*numTetaSmplPerPos);
        qdB = qrx1B_r_tx2(1+(i-1)*numTetaSmplPerPos:i*numTetaSmplPerPos);
        qdC = qrx1C_r_tx2(1+(i-1)*numTetaSmplPerPos:i*numTetaSmplPerPos);
        qdD = qrx1D_r_tx2(1+(i-1)*numTetaSmplPerPos:i*numTetaSmplPerPos);
        actwav =  tx2_wav(1+(i-1)*numTetaSmplPerPos:i*numTetaSmplPerPos);
        [~, calc_aoa_qrx1_tx2(i), ~, b] = vlpAlgoSim_getTeta( qrx.f_QRX.map_inv, qdA, qdB, qdC, qdD, actwav, validAoAThd);
        vldFlag_qrx1 = a*b;
        
        % Algo: Find theta, QRX 2
        qdA = qrx2A_r_tx1(1+(i-1)*numTetaSmplPerPos:i*numTetaSmplPerPos);
        qdB = qrx2B_r_tx1(1+(i-1)*numTetaSmplPerPos:i*numTetaSmplPerPos);
        qdC = qrx2C_r_tx1(1+(i-1)*numTetaSmplPerPos:i*numTetaSmplPerPos);
        qdD = qrx2D_r_tx1(1+(i-1)*numTetaSmplPerPos:i*numTetaSmplPerPos);
        actwav = tx1_wav(1+(i-1)*numTetaSmplPerPos:i*numTetaSmplPerPos);
        [~, calc_aoa_qrx2_tx1(i), ~, a] = vlpAlgoSim_getTeta( qrx.f_QRX.map_inv, qdA, qdB, qdC, qdD, actwav, validAoAThd);
        qdA = qrx2A_r_tx2(1+(i-1)*numTetaSmplPerPos:i*numTetaSmplPerPos);
        qdB = qrx2B_r_tx2(1+(i-1)*numTetaSmplPerPos:i*numTetaSmplPerPos);
        qdC = qrx2C_r_tx2(1+(i-1)*numTetaSmplPerPos:i*numTetaSmplPerPos);
        qdD = qrx2D_r_tx2(1+(i-1)*numTetaSmplPerPos:i*numTetaSmplPerPos);
        actwav =  tx2_wav(1+(i-1)*numTetaSmplPerPos:i*numTetaSmplPerPos);
        [~, calc_aoa_qrx2_tx2(i), ~, b] = vlpAlgoSim_getTeta( qrx.f_QRX.map_inv, qdA, qdB, qdC, qdD, actwav, validAoAThd);
        vldFlag_qrx2 = a*b;
        
        % When the target is out of the FoV, obviously it's an invalid result.
        % we detect this in the function and assign a NaN to those results here
        % so that they don't appear on the plot. It's just extra clutter when
        % they're plotted.
        if( (vldFlag_qrx1 == 0) || (vldFlag_qrx2 == 0))
            calc_x11(i) = NaN;
            calc_y11(i) = NaN;
            calc_x12(i) = NaN;
            calc_y12(i) = NaN;
        else
            % Algo: find position, use sine law!
            L = vehicle.ego.width;
            calc_x11(i) = L*(1 + sind(calc_aoa_qrx2_tx1(i))*cosd(calc_aoa_qrx1_tx1(i))/sind(calc_aoa_qrx1_tx1(i)-calc_aoa_qrx2_tx1(i)));
            calc_y11(i) = L*cosd(calc_aoa_qrx2_tx1(i))*cosd(calc_aoa_qrx1_tx1(i))/sind(calc_aoa_qrx1_tx1(i)-calc_aoa_qrx2_tx1(i));
            calc_x12(i) = L*(1 + sind(calc_aoa_qrx2_tx2(i))*cosd(calc_aoa_qrx1_tx2(i))/sind(calc_aoa_qrx1_tx2(i)-calc_aoa_qrx2_tx2(i)));
            calc_y12(i) = L*cosd(calc_aoa_qrx2_tx2(i))*cosd(calc_aoa_qrx1_tx2(i))/sind(calc_aoa_qrx1_tx2(i)-calc_aoa_qrx2_tx2(i));
        end
        
    end
    
    cx11(:,iter) = calc_x11;
    cy11(:,iter) = calc_y11;
    cx12(:,iter) = calc_x12;
    cy12(:,iter) = calc_y12;
    aoa_11(:,iter) = calc_aoa_qrx1_tx1;
    aoa_12(:,iter) = calc_aoa_qrx1_tx2;
    aoa_21(:,iter) = calc_aoa_qrx2_tx1;
    aoa_22(:,iter) = calc_aoa_qrx2_tx2;
end
close(fii)

save(res_file,'vehicle','cx11','cy11','cx12','cy12','aoa_11','aoa_12','aoa_21','aoa_22','vlpDecimRate');


end