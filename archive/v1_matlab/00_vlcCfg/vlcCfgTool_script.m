clear all; close all; clc;
addpath('fcn')

%% QRX + TIA setup

%%% Note: the setup in the following article is used as "becha":
% Béchadergue, B., Chassagne, L., & Guan, H. (2017, October). 
% A visible light-based system for automotive relative positioning. 
% In 2017 IEEE SENSORS (pp. 1-3). IEEE.
%
% We refer to this as "becha" below. They have an interesting way of
% defining a receiver using only the active area and the FoV. This can be 
% interpreted as a photodetector with an aperture over it to limit the FoV
% to that stated value. We modified our formulation to match this by taking
% the lens area as the "active" area, and scaling the photodetector
% accordingly. Therefore, we use 6.3mm as d_H, which corresponds to an
% active area of ~39.69mm^2. We scaled the capacitance accordingly, using
% the 112 pf/cm^2 value provided in becha. However, note that the
% difference in capacitance values causes very small changes due to the
% fact that the operation is shot-noise-limited rather than thermal noise,
% and the capacitance only plays a role in thermal noise on the TIA.

% Optical: Parameters
qrx.f_QRX.params.d_L   = 07.10e-3; % [m]                Modify this, opr -> 03.00e-3 , becha -> 07.10e-3 
qrx.f_QRX.params.d_H   = 06.30e-3; % [m]                Modify this, opr -> 02.79e-3 , becha -> 06.30e-3
qrx.f_QRX.params.d_X   = 00.55e-3; % [m]                Modify this, opr -> 00.25e-3 , becha -> 00.55e-3
qrx.f_QRX.params.n     = 01.50;    % []                 Modify this, opr -> 01.50    , becha -> 01.50
qrx.f_QRX.params.d_F   = qrx.f_QRX.params.d_L/qrx.f_QRX.params.n; % [m]
qrx.f_QRX.params.area  = qrx.f_QRX.params.d_H.^2;   % [m^2], total area for 4 quadrants

% Optical: Resolutions for LUT generation
qrx.f_QRX.params.simRes_d     = 01.00e-6; % [m]         Modify this
qrx.f_QRX.params.simRes_theta = 05.00e-2; % [deg]       Modify this

% Optical: Generate g_QRX and add it to the structure
[ f_QRX_map, g_QRX_map, fov_lim_neg, fov_lim_pos ] = vlcCfgTool_qrxMap( qrx.f_QRX.params );
qrx.f_QRX.map        = f_QRX_map; 
qrx.f_QRX.map_inv    = g_QRX_map; 
qrx.f_QRX.fovLimits  = [ fov_lim_neg, fov_lim_pos ];
disp(sprintf('feval at 0: %f ', feval(f_QRX_map,0)))
clear g_QRX_map fov_lim_neg fov_lim_pos

% Electrical: Parameters
qrx.tia.params.electron  = 01.60e-19;      % [C]        Modify this
qrx.tia.params.boltzmann = 01.38e-23;      % [J/K]      Modify this
qrx.tia.params.gamma     = 00.50;          % [A/W]      Modify this, opr -> 00.45    , becha -> 00.50
qrx.tia.params.gm        = 30.00e-3;       % [S]        Modify this
qrx.tia.params.bandwidth = 10.00e6;        % [Hz]       Modify this
qrx.tia.params.C_T       = 44.45e-12;      % [F]        Modify this, opr -> 40.00e-12, becha -> (112e-12)*0.3969 = 44.45e-12
qrx.tia.params.R_F       = 02.84e3;        % [Ohm]      Modify this, opr -> 10.00e3  , becha -> 02.84e3, computed from G, see our article
qrx.tia.params.capGamma  = 01.50;          % []         Modify this
qrx.tia.params.I_B2      = 05.62e-1;       % []         Modify this
qrx.tia.params.I_B3      = 08.68e-2;       % []         Modify this

% Electrical: noise factors
q = qrx.tia.params.electron;              % temporary
k = qrx.tia.params.boltzmann;             % temporary
gamma = qrx.tia.params.gamma;             % temporary
gm = qrx.tia.params.gm;                   % temporary
B = qrx.tia.params.bandwidth;             % temporary
cptGamma = qrx.tia.params.capGamma;       % temporary
i2 = qrx.tia.params.I_B2;                 % temporary
i3 = qrx.tia.params.I_B3;                 % temporary
C_T = qrx.tia.params.C_T;                 % temporary
R_F = qrx.tia.params.R_F;                 % temporary
qrx.tia.shot_P_r_factor   = 2*q*gamma*B;                               % [A^2/W] -> [A^2 when *P_r]
qrx.tia.shot_I_bg_factor  = 2*q*i2*B;                                  % [A]     -> [A^2 when *I_bg]
qrx.tia.thermal_factor1   = 4*k*i2*B/R_F;                              % [A^2/K] -> [A^2 when *T]
qrx.tia.thermal_factor2   = 4*k*i3*(B.^3)*cptGamma*((2*pi*C_T).^2)/gm; % [A^2/K] -> [A^2 when *T]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Small debugging test for noise %%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% P_r  = 01.00e-3; % [W]
% I_bg = 74.00e-4; % [A]
% T    = 298;      % [K]
% var_shot    = qrx.tia.shot_P_r_factor*P_r + qrx.tia.shot_I_bg_factor*I_bg;
% var_thermal = T*(qrx.tia.thermal_factor1 + qrx.tia.thermal_factor2);
% stdev_noise = sqrt(var_shot + var_thermal)*R_F;
% disp(sprintf('Temperature:  %d K', T))
% disp(sprintf('RX Power:     %f mW', P_r*1e3))
% disp(sprintf('Background I: %f uA', I_bg*1e6))
% disp(sprintf('Stdev in mV:  %f mV', stdev_noise*1e3))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear q k gm gamma B cptGamma i2 i3 C_T R_F var_shot var_thermal stdev_noise I_bg P_r T


%% TX setup

%%% Note: the setup in the abovementioned article is used for the 
%         Lambertian approximation TX
%
% The becha article uses 2 headlights (2 W) and 1 taillight (1 W) for
% positioning 1 taillight, and they need 2 for localiztion. This is 
% obviously much higher than what we need, i.e., only 2 taillights. To 
% balance the conditions while still staying inside limits put forth by 
% the ECE regulations for the rear brake/stop lights, we used two 2 W
% taillights. We complied with the Lambertian asssumption for the time
% being although we acknowledge that that's not an accurate representation
% of the beam patterns discussed in the ECE regulations. That's future
% work...

% Optical: normalized beam pattern, 
tx.pattern.params.simRes_theta = 01.00e-1; % [deg]       Modify this

%%% Lambertian pattern, radially symmetric
tx.pattern.params.half_angle = 20;         % [deg]       Modify this
[ tx_map ] = vlcCfgTool_txMap(tx.pattern.params, 'lambertian_radSym', 60);

% %%% Custom pattern, radially symmetric
% tx.pattern.params.num_pattern_fn = 'data/customPattern_fogTX.mat';
% [ tx_map ] = vlcCfgTool_txMap(tx.pattern.params, 'custom_radSym', 60);

%%% Custom pattern, not radially symmetric (future work, ECE-compliant)
% tx.pattern.params.num_pattern_fn = '...';
% [ tx_map ] = vlcCfgTool_txMap(tx.pattern.params, 'custom_notSym', 60);

tx.pattern.map = tx_map;

% Electrical: power
tx.power = 2; %[W], total optical power for the normalized pattern.

clear m angle_array half_angle tx_map
%% Save Config
answer = inputdlg('Enter filename for the config file','Config Filename',[1 50],{'vlcCfg_<explanation>.mat'});
save(strcat('data/',answer{1}),'tx','qrx')
