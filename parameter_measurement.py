import numpy as np

#######################################################################################
### aoa measurement  -> sonercoleri
def measure_angle_fQRX(fqrx_samples, actual_phi):
    fqrx_values   = fqrx_samples[:,1]
    diff_value    = np.abs(fqrx_values - actual_phi)
    argpart_value = np.argpartition(diff_value, 2) # , axis = 0

    closest_value_0_idx = argpart_value[0] # index for closest y value in LUT
    closest_value_1_idx = argpart_value[1] # index for 2nd closest y value in LUT

    closest_value_0 = fqrx_values[closest_value_0_idx] # closest y value in LUT
    closest_value_1 = fqrx_values[closest_value_1_idx] # 2nd closest y value in LUT

    fqrx_angles = fqrx_samples[:,0];
    closest_angle_0 = fqrx_angles[closest_value_0_idx] # angle value for closest y value in LUT
    closest_angle_1 = fqrx_angles[closest_value_1_idx] # angle value for 2nd closest y value in LUT

    value_dist_to_0 = actual_phi - closest_value_0; # increment

    angle_interval = closest_angle_1 - closest_angle_0;  # total span between two angle values
    value_interval = closest_value_1 - closest_value_0; # total span between two y values
    slope          = angle_interval / value_interval;
    angle_est      = value_dist_to_0 * slope + closest_angle_0;
    return angle_est

def aoa_measurement(sigA_buffer, sigB_buffer, sigC_buffer, sigD_buffer, wav_buffer, f_QRX, thd):
    qA = np.abs(np.mean(sigA_buffer*wav_buffer));
    qB = np.abs(np.mean(sigB_buffer*wav_buffer));
    qC = np.abs(np.mean(sigC_buffer*wav_buffer));
    qD = np.abs(np.mean(sigD_buffer*wav_buffer));
    qpwr = qA + qB + qC + qD;
    
    # When the target is out of the FoV, obviously it's an invalid result.
    # we detect this here and assign a NaN to those results outside
    # so that they don't appear on the plot. It's just extra clutter when
    # they're plotted.
    if( ((qC + qD) < thd) or ((qA + qB) < thd) ):
        aoa = 0;
    else:
        phi_hrz_est = ((qC+qD)-(qA+qB))/qpwr;
        aoa = measure_angle_fQRX(f_QRX, phi_hrz_est);

    return aoa

#######################################################################################
### pdoa measurement -> roberts

#######################################################################################
### rtof measurement -> bechadergue
