function [ theta_horizontal, validFlag ] = aoa_extract_theta(qrx, sigA, sigB, sigC, sigD, wav, thd)
%aoa2_extract_theta_ij Summary of this function goes here
%   Detailed explanation goes here

qA = abs(mean(sigA.*wav));
qB = abs(mean(sigB.*wav));
qC = abs(mean(sigC.*wav));
qD = abs(mean(sigD.*wav));
q_pwr = qA + qB + qC + qD;

% When the target is out of the FoV, obviously it's an invalid result.
% we detect this here and assign a NaN to those results outside
% so that they don't appear on the plot. It's just extra clutter when
% they're plotted.
if( ((qB+qD)<thd) || ((qA+qC)<thd) )
    validFlag = 0;
    theta_horizontal = 0;
%     theta_vertical = 0;
else
    validFlag = 1;
    phi_x_est = ((qB+qD)-(qA+qC))/(q_pwr); % x 180deg ters tanimlandiydi
    
    %%% New method
    theta_horizontal = feval(qrx.f_QRX.map_inv,phi_x_est);
%     vertical angle measurement not used!
end


end

