function [ pwr, teta_hrz, teta_vrt, validFlag ] = vlpAlgoSim_getTeta( lutto, qdA_volts, qdB_volts, qdC_volts, qdD_volts, actWav, thd)
%VLP_ALGO_TETACALC Summary of this function goes here
%   Detailed explanation goes here

% Demodulation with original TX signal, see system model assumptions

% Old 
qdA_energy = abs(mean(qdA_volts.*actWav));
qdB_energy = abs(mean(qdB_volts.*actWav));
qdC_energy = abs(mean(qdC_volts.*actWav));
qdD_energy = abs(mean(qdD_volts.*actWav));

pwr = qdA_energy + qdB_energy + qdC_energy + qdD_energy;

% When the target is out of the FoV, obviously it's an invalid result.
% we detect this here and assign a NaN to those results outside
% so that they don't appear on the plot. It's just extra clutter when
% they're plotted.
if( ((qdB_energy+qdD_energy)<thd) || ((qdA_energy+qdC_energy)<thd) )
	validFlag = 0;
	teta_hrz = 0;
	teta_vrt = 0;
else
	validFlag = 1;

    phi_x_est = ((qdB_energy+qdD_energy)-(qdA_energy+qdC_energy))/(pwr); % x 180deg ters tanimlandiydi
    phi_y_est = ((qdA_energy+qdB_energy)-(qdC_energy+qdD_energy))/(pwr);

	%%% Old method
	% idx0 = find(abs(lut(2,:)-x_est)==min(abs(lut(2,:)-x_est)));
	% idy0 = find(abs(lut(2,:)-y_est)==min(abs(lut(2,:)-y_est)));
	% teta_hrz = lut(1,idx0(1));
	% teta_vrt = lut(1,idy0(1));

	%%% New method
	teta_hrz = feval(lutto,phi_x_est);%-feval(lutto,0); % -0.031 offset required
	teta_vrt = feval(lutto,phi_y_est);%-feval(lutto,0); % -0.031 offset required

    % vertical angle measurement not used!
end

end

