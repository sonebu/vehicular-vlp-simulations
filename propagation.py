import numpy as np
from numba import njit
import time

##############################################################################
### propagation delay

def propagation_delay(x,y,c):
    # we ignore z
    delay = np.sqrt(x**2 + y**2)/c;
    return delay


##############################################################################
### Method 1 - Approximate the 3D Lambertian pattern with a rectangular prism  
###            for very small angular intervals (fast, but biased/approximate)
def tx_solidangles(x, y, z, heading, dim):
    p1_xy = np.arctan2(y,x - dim/2);
    p2_xy = np.arctan2(y,x + dim/2);
    e1_xy = np.rad2deg((np.deg2rad(90) - p1_xy - np.deg2rad(heading)));
    e2_xy = np.rad2deg((np.deg2rad(90) - p2_xy - np.deg2rad(heading)));

    p1_xz = np.arctan2(y,z - dim/2);
    p2_xz = np.arctan2(y,z + dim/2);
    e1_xz = np.rad2deg((np.deg2rad(90) - p1_xz)) # - np.deg2rad(pitch)));, flat roads
    e2_xz = np.rad2deg((np.deg2rad(90) - p2_xz)) # - np.deg2rad(pitch)));, flat roads
    
    return e1_xy, e2_xy, e1_xz, e2_xz

def received_power(x, y, z, dim, hdg, tx_pwr, tx_norm, tx_lambertian_order, attenuation_factor):
    # the portion of the lambertian pattern staying within these angles is a rectangular prism with a "wavy" top. 
    # prism is smooth though, + we always check veeery small solidangle portions, so we can assume
    # the top is close to being a linear slump that is so smooth, it's nearly a rectangular prism
    # with height = average height of the four corners. This should bring very small error. 
    angles = tx_solidangles(x, y, z, hdg, dim)
        
    # since the pattern is radially symmetric, we can keep just one (x,y) pattern, 
    # where the x values can be computed as such, using xy and zy angles (think Pythagorean):
    pattern_angle_xy1_zy1 = np.sqrt(angles[0]**2 + angles[2]**2)
    pattern_angle_xy1_zy2 = np.sqrt(angles[0]**2 + angles[3]**2)
    pattern_angle_xy2_zy1 = np.sqrt(angles[1]**2 + angles[2]**2)
    pattern_angle_xy2_zy2 = np.sqrt(angles[1]**2 + angles[3]**2)
    
    val_xy1_zy1 = np.cos(np.deg2rad(pattern_angle_xy1_zy1))**tx_lambertian_order # lambertian
    val_xy1_zy2 = np.cos(np.deg2rad(pattern_angle_xy1_zy2))**tx_lambertian_order # lambertian
    val_xy2_zy1 = np.cos(np.deg2rad(pattern_angle_xy2_zy1))**tx_lambertian_order # lambertian
    val_xy2_zy2 = np.cos(np.deg2rad(pattern_angle_xy2_zy2))**tx_lambertian_order # lambertian
    
    # length and width of the prism
    angle_dist_xy = np.abs(angles[0] - angles[1])
    angle_dist_zy = np.abs(angles[2] - angles[3])

    # average height of the prism
    avg_val = (val_xy1_zy1 + val_xy1_zy2 + val_xy2_zy1 + val_xy2_zy2)/4 # average height of that rectangular prism with wavy top
    
    pwr = tx_pwr*angle_dist_xy*angle_dist_zy*avg_val/tx_norm;

    # on top of this, we need to apply weather-dependent attenuation
    tx_rx_distance = np.sqrt(x**2 + y**2)
    attenuation    = 10**(tx_rx_distance*attenuation_factor/10);
    pwr = pwr*attenuation

    return pwr

def received_power_nonlambertian(x, y, z, dim, hdg, tx_pwr, tx_norm, tx_pattern, tx_thetaarray, tx_phiarray, attenuation_factor):
    # the portion of the lambertian pattern staying within these angles is a rectangular prism with a "wavy" top. 
    # prism is smooth though, + we always check veeery small solidangle portions, so we can assume
    # the top is close to being a linear slump that is so smooth, it's nearly a rectangular prism
    # with height = average height of the four corners. This should bring very small error. 
    eps1_xy, eps2_xy, eps1_zy, eps2_zy = tx_solidangles(x, y, z, hdg, dim)

    val = asymmetricSrc3dIntegral_smallangle(tx_pattern, tx_thetaarray, tx_phiarray, 
                                             eps1_xy, eps2_xy, eps1_zy, eps2_zy)

    pwr = tx_pwr*val/tx_norm;

    # on top of this, we need to apply weather-dependent attenuation
    tx_rx_distance = np.sqrt(x**2 + y**2)
    attenuation    = 10**(tx_rx_distance*attenuation_factor/10);
    pwr = pwr*attenuation

    return pwr

##############################################################################
### Method 2 - Numerical integration (very slow, but asymptotically unbiased) 

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def radSymSrc3dIntegral( rad_pat, eps1_xy, eps2_xy, eps1_zy, eps2_zy ):
    id0_eps1_xy = find_nearest(rad_pat[0,:], eps1_xy);
    id0_eps2_xy = find_nearest(rad_pat[0,:], eps2_xy);
    id0_eps1_zy = find_nearest(rad_pat[0,:], eps1_zy);
    id0_eps2_zy = find_nearest(rad_pat[0,:], eps2_zy);

    id_zero = int(rad_pat.shape[1]/2); # works because python is 0 indexed
    
    id_list_numsamples_xy = int(abs(id0_eps2_xy-id0_eps1_xy)+1);
    id_list_numsamples_zy = int(abs(id0_eps2_zy-id0_eps1_zy)+1);
    
    id_list_fwd_xy = np.linspace(id0_eps2_xy, id0_eps1_xy, num=id_list_numsamples_xy);
    id_list_bwd_xy = np.linspace(id0_eps1_xy, id0_eps2_xy, num=id_list_numsamples_xy);
    id_list_fwd_zy = np.linspace(id0_eps2_zy, id0_eps1_zy, num=id_list_numsamples_zy);
    id_list_bwd_zy = np.linspace(id0_eps1_zy, id0_eps2_zy, num=id_list_numsamples_zy);
    
    # note that the same rectangular prism of average height approximation in method1 is also done below, 
    # but here it's done on a sample-by-sample basis, not for an angle interval. So here it is done 
    # to sort of over-sample the LUT that we generate using the pattern expression.

    vol_fwd = 0;
    for i in range(1, id_list_numsamples_xy): # i=2:length(id_list_fwd_xy)
        for j in range(1, id_list_numsamples_zy): # j=2:length(id_list_fwd_zy)
            radial_pos_i_xy_i_zy     = int(np.sqrt((id_list_fwd_xy[i] - id_zero)**2 + (id_list_fwd_zy[j] - id_zero)**2))
            radial_pos_im1_xy_i_zy   = radial_pos_i_xy_i_zy-1; # this is an approximation, while this should really go into the expression above, since that would not hit a different LUT value for very small angles, we do this
            radial_pos_i_xy_im1_zy   = radial_pos_i_xy_i_zy-1; # this is an approximation, while this should really go into the expression above, since that would not hit a different LUT value for very small angles, we do this
            radial_pos_im1_xy_im1_zy = radial_pos_i_xy_i_zy-2; # this is an approximation, while this should really go into the expression above, since that would not hit a different LUT value for very small angles, we do this
            avg_y_val = (rad_pat[1,radial_pos_i_xy_i_zy+id_zero] + \
                rad_pat[1,radial_pos_im1_xy_i_zy+id_zero] + \
                rad_pat[1,radial_pos_i_xy_im1_zy+id_zero] + \
                rad_pat[1,radial_pos_im1_xy_im1_zy+id_zero] )/4 ;
            vol_fwd = vol_fwd + (rad_pat[0,radial_pos_i_xy_i_zy+id_zero]-rad_pat[0,radial_pos_im1_xy_i_zy+id_zero])* \
                (rad_pat[0,radial_pos_i_xy_i_zy+id_zero]-rad_pat[0,radial_pos_i_xy_im1_zy+id_zero])* \
                avg_y_val;

    vol_bwd = 0;
    for i in range(1, id_list_numsamples_xy): # i=2:length(id_list_bwd_xy)
        for j in range(1, id_list_numsamples_zy): # j=2:length(id_list_bwd_zy)
            radial_pos_i_xy_i_zy     = int(np.sqrt((id_list_bwd_xy[i] - id_zero)**2 + (id_list_bwd_zy[j] - id_zero)**2))
            radial_pos_im1_xy_i_zy   = radial_pos_i_xy_i_zy-1; # this is an approximation, while this should really go into the expression above, since that would not hit a different LUT value for very small angles, we do this
            radial_pos_i_xy_im1_zy   = radial_pos_i_xy_i_zy-1; # this is an approximation, while this should really go into the expression above, since that would not hit a different LUT value for very small angles, we do this
            radial_pos_im1_xy_im1_zy = radial_pos_i_xy_i_zy-2; # this is an approximation, while this should really go into the expression above, since that would not hit a different LUT value for very small angles, we do this
            avg_y_val = (rad_pat[1,radial_pos_i_xy_i_zy+id_zero] + \
                rad_pat[1,radial_pos_im1_xy_i_zy+id_zero] + \
                rad_pat[1,radial_pos_i_xy_im1_zy+id_zero] + \
                rad_pat[1,radial_pos_im1_xy_im1_zy+id_zero] )/4 ;
            vol_bwd = vol_bwd + (rad_pat[0,radial_pos_i_xy_i_zy+id_zero]-rad_pat[0,radial_pos_im1_xy_i_zy+id_zero])* \
                (rad_pat[0,radial_pos_i_xy_i_zy+id_zero]-rad_pat[0,radial_pos_i_xy_im1_zy+id_zero])* \
                avg_y_val;

    # we compensate the above approximations by averaging forward and backward passes, 
    # based on the fact that the curve we're doing this on is very smooth. So just improving
    # the oversampling process with an average over neighbor samples (and their gradients actually).
    vol = (vol_fwd+vol_bwd)/2;
    return vol

##############################################################################
### Method 4 - Numerical integration for asymmetric patterns, see optics/utlities for more info

# ref: https://stackoverflow.com/a/23734295
@njit(parallel=True, fastmath=True)
def find_k_nearest(array, value, k):
    array = np.asarray(array)
    vals  = np.abs(array - value)
    idx   = vals.argsort()[:k]
    #idx   = idx[np.argsort(vals[idx])]
    return idx

@njit(parallel=True, fastmath=True)
def getinterp(y2,y1,x2,x1,xnew):
    return (xnew-x1)*(y2-y1)/(x2-x1)+y1

@njit(parallel=True, fastmath=True)
def asymmetricSrc3dIntegral_smallangle( pat, theta_anglearray, phi_anglearray, eps1_xy, eps2_xy, eps1_zy, eps2_zy ):
    ids_eps1_xy   = find_k_nearest(theta_anglearray, eps1_xy, 2) # we get 2 values since an exact hit is improbable
    idtop_eps1_xy = ids_eps1_xy.max() # closest top point
    idbot_eps1_xy = ids_eps1_xy.min() # closest bottom point
    eps1_xy_top = theta_anglearray[idtop_eps1_xy]
    eps1_xy_bot = theta_anglearray[idbot_eps1_xy]
    
    # same for other 3 quantities
    ids_eps2_xy   = find_k_nearest(theta_anglearray, eps2_xy, 2);
    idtop_eps2_xy = ids_eps2_xy.max()
    idbot_eps2_xy = ids_eps2_xy.min()
    eps2_xy_top = theta_anglearray[idtop_eps2_xy]
    eps2_xy_bot = theta_anglearray[idbot_eps2_xy]
    
    ids_eps1_zy   = find_k_nearest(phi_anglearray, eps1_zy, 2);
    idtop_eps1_zy = ids_eps1_zy.max()
    idbot_eps1_zy = ids_eps1_zy.min()
    eps1_zy_top = phi_anglearray[idtop_eps1_zy]
    eps1_zy_bot = phi_anglearray[idbot_eps1_zy]
    
    ids_eps2_zy   = find_k_nearest(phi_anglearray, eps2_zy, 2);
    idtop_eps2_zy = ids_eps2_zy.max()
    idbot_eps2_zy = ids_eps2_zy.min()
    eps2_zy_top = phi_anglearray[idtop_eps2_zy]
    eps2_zy_bot = phi_anglearray[idbot_eps2_zy]
    
    # first compute the interp heights on the two sides of the plane,
    # then use their height vals for a final interp over the other axis.
    eps1xy1zy_z_bot_bot = pat[idbot_eps1_xy, idbot_eps1_zy]
    eps1xy1zy_z_bot_top = pat[idbot_eps1_xy, idtop_eps1_zy]
    eps1xy1zy_z_top_bot = pat[idtop_eps1_xy, idbot_eps1_zy]
    eps1xy1zy_z_top_top = pat[idtop_eps1_xy, idtop_eps1_zy]
    eps1xy1zy_z_interp_right = getinterp(eps1xy1zy_z_top_top, eps1xy1zy_z_top_bot, 
                                         eps1_zy_top, eps1_zy_bot, eps1_zy)
    eps1xy1zy_z_interp_left  = getinterp(eps1xy1zy_z_bot_top, eps1xy1zy_z_bot_bot, 
                                         eps1_zy_top, eps1_zy_bot, eps1_zy)
    eps1xy1zy_z_interp_final = getinterp(eps1xy1zy_z_interp_right, eps1xy1zy_z_interp_left, 
                                         eps1_xy_top, eps1_xy_bot, eps1_xy)

    # repeat for the other 3 eps points  
    eps1xy2zy_z_bot_bot = pat[idbot_eps1_xy, idbot_eps2_zy]
    eps1xy2zy_z_bot_top = pat[idbot_eps1_xy, idtop_eps2_zy]
    eps1xy2zy_z_top_bot = pat[idtop_eps1_xy, idbot_eps2_zy]
    eps1xy2zy_z_top_top = pat[idtop_eps1_xy, idtop_eps2_zy]
    eps1xy2zy_z_interp_right = getinterp(eps1xy2zy_z_top_top, eps1xy2zy_z_top_bot, 
                                         eps2_zy_top, eps2_zy_bot, eps2_zy)
    eps1xy2zy_z_interp_left  = getinterp(eps1xy2zy_z_bot_top, eps1xy2zy_z_bot_bot, 
                                         eps2_zy_top, eps2_zy_bot, eps2_zy)
    eps1xy2zy_z_interp_final = getinterp(eps1xy2zy_z_interp_right, eps1xy2zy_z_interp_left, 
                                         eps1_xy_top, eps1_xy_bot, eps1_xy)

    eps2xy1zy_z_bot_bot = pat[idbot_eps2_xy, idbot_eps1_zy]
    eps2xy1zy_z_bot_top = pat[idbot_eps2_xy, idtop_eps1_zy]
    eps2xy1zy_z_top_bot = pat[idtop_eps2_xy, idbot_eps1_zy]
    eps2xy1zy_z_top_top = pat[idtop_eps2_xy, idtop_eps1_zy]
    eps2xy1zy_z_interp_right = getinterp(eps2xy1zy_z_top_top, eps2xy1zy_z_top_bot, 
                                         eps1_zy_top, eps1_zy_bot, eps1_zy)
    eps2xy1zy_z_interp_left  = getinterp(eps2xy1zy_z_bot_top, eps2xy1zy_z_bot_bot, 
                                         eps1_zy_top, eps1_zy_bot, eps1_zy)
    eps2xy1zy_z_interp_final = getinterp(eps2xy1zy_z_interp_right, eps2xy1zy_z_interp_left, 
                                         eps2_xy_top, eps2_xy_bot, eps2_xy)
    
    eps2xy2zy_z_bot_bot = pat[idbot_eps2_xy, idbot_eps2_zy]
    eps2xy2zy_z_bot_top = pat[idbot_eps2_xy, idtop_eps2_zy]
    eps2xy2zy_z_top_bot = pat[idtop_eps2_xy, idbot_eps2_zy]
    eps2xy2zy_z_top_top = pat[idtop_eps2_xy, idtop_eps2_zy]
    eps2xy2zy_z_interp_right = getinterp(eps2xy2zy_z_top_top, eps2xy2zy_z_top_bot, 
                                         eps2_zy_top, eps2_zy_bot, eps2_zy)
    eps2xy2zy_z_interp_left  = getinterp(eps2xy2zy_z_bot_top, eps2xy2zy_z_bot_bot, 
                                         eps2_zy_top, eps2_zy_bot, eps2_zy)
    eps2xy2zy_z_interp_final = getinterp(eps2xy2zy_z_interp_right, eps2xy2zy_z_interp_left, 
                                         eps2_xy_top, eps2_xy_bot, eps2_xy)
    
    vol = (eps2_xy - eps1_xy)*(eps2_zy - eps1_zy) * \
          (eps1xy1zy_z_interp_final + eps1xy2zy_z_interp_final +\
           eps2xy1zy_z_interp_final + eps2xy2zy_z_interp_final)/4
    
    return vol


##############################################################################
### Quad detector function fit

def find_closest_linear_fit(fqrx_samples, actual_angle):
    fqrx_samples_rep = np.expand_dims(fqrx_samples, axis=0)

    # sadece tek eleman olunca "neyi repeat edeyim" diye hata veriyor.
    # tek eleman da buraya boş tuple olarak geliyor o yüzden şunu koyduk -> ()
    if(actual_angle.shape != ()):
        fqrx_samples_rep = np.repeat(fqrx_samples_rep, actual_angle.shape[0], axis=0)
        actual_angle_exp = np.expand_dims(actual_angle, axis=1)
    else:
        actual_angle_exp = actual_angle;

    fqrx_angles_rep = fqrx_samples_rep[:,:,0]
    diff_angle    = np.abs(fqrx_angles_rep - actual_angle_exp)
    argpart_angle = np.argpartition(diff_angle, 2, axis = 1) # axis=1 because it's batched!

    closest_angle_0_idx = argpart_angle[:,0] # index for closest angle value in LUT
    closest_angle_1_idx = argpart_angle[:,1] # index for 2nd closest angle value in LUT

    closest_angle_0 = fqrx_angles_rep[range(0,closest_angle_0_idx.shape[0]), closest_angle_0_idx] # closest angle value in LUT
    closest_angle_1 = fqrx_angles_rep[range(0,closest_angle_1_idx.shape[0]), closest_angle_1_idx] # 2nd closest angle value in LUT

    fqrx_values_rep = fqrx_samples_rep[:,:,1];
    closest_val_0 = fqrx_values_rep[range(0,closest_angle_0_idx.shape[0]), closest_angle_0_idx] # y value for closest angle value in LUT
    closest_val_1 = fqrx_values_rep[range(0,closest_angle_1_idx.shape[0]), closest_angle_1_idx] # y value for 2nd closest angle value in LUT

    angle_dist_to_0 = actual_angle - closest_angle_0; # increment

    val_interval   = closest_val_1 - closest_val_0;     # total span between two y values
    angle_interval = closest_angle_1 - closest_angle_0; # total span between two angle values
    slope          = val_interval / angle_interval;
    val            = angle_dist_to_0 * slope + closest_val_0;
    return val

def quad_shares_power(x,y,z,fqrx):
    teta_v = np.rad2deg(np.arctan2(z,y)); # not used
    teta_h = np.rad2deg(np.arctan2(x,y));
    phi = find_closest_linear_fit(fqrx, teta_h);
    B_sh = np.minimum(np.maximum((phi+1)/2, np.zeros_like(phi)-1), np.zeros_like(phi)+1);
    D_sh = np.minimum(np.maximum((phi+1)/2, np.zeros_like(phi)-1), np.zeros_like(phi)+1);
    A_sh = 1-B_sh;
    C_sh = 1-D_sh;

    A_sh = np.expand_dims(A_sh/2, axis=1);
    B_sh = np.expand_dims(B_sh/2, axis=1);
    C_sh = np.expand_dims(C_sh/2, axis=1);
    D_sh = np.expand_dims(D_sh/2, axis=1);

    return np.concatenate((A_sh, B_sh, C_sh, D_sh), axis=1)
    
def quad_distribute_power(x,y,z,fqrx, power):
    shares = quad_shares_power(x,y,z,fqrx);
    shares_pwr = np.zeros_like(shares)
    shares_pwr[:,0] = power*shares[:,0];
    shares_pwr[:,1] = power*shares[:,1];
    shares_pwr[:,2] = power*shares[:,2];
    shares_pwr[:,3] = power*shares[:,3];
    return shares_pwr

##############################################################################
### generating sinusoidal rx signals from qrx

def gen_qrx_onlyclocked(shared_pwr_txL_to_rxL, shared_pwr_txL_to_rxR, shared_pwr_txR_to_rxL, shared_pwr_txR_to_rxR,
                        delay_txL_to_rxL, delay_txL_to_rxR, delay_txR_to_rxL, delay_txR_to_rxR,
                        f_eL, f_eR, pd_snst, pd_gain, thermal_and_bg_curr, rx_P_rx_factor,
                        step_time, simulation_time, smp_lo, smp_hi,
                        add_noise):
    ### for simplicity, let's just assume a simple sine wave for each TX. 
    ### Roberts wants 10-50 MHz with higher=better, but that's not feasible. 
    ### Bechadergue ideally wants something on the order of MHz, and SonerColeri doesn't care, 
    ### so let's stick to 1 MHz / 0.9 MHz, which is feasible

    ### noiseless waveforms first
    rxLA_txL_peakAmps = shared_pwr_txL_to_rxL[smp_lo:smp_hi,0]*pd_snst
    rxLC_txL_peakAmps = shared_pwr_txL_to_rxL[smp_lo:smp_hi,1]*pd_snst
    rxLB_txL_peakAmps = shared_pwr_txL_to_rxL[smp_lo:smp_hi,2]*pd_snst
    rxLD_txL_peakAmps = shared_pwr_txL_to_rxL[smp_lo:smp_hi,3]*pd_snst
    rxRA_txL_peakAmps = shared_pwr_txL_to_rxR[smp_lo:smp_hi,0]*pd_snst
    rxRC_txL_peakAmps = shared_pwr_txL_to_rxR[smp_lo:smp_hi,1]*pd_snst
    rxRB_txL_peakAmps = shared_pwr_txL_to_rxR[smp_lo:smp_hi,2]*pd_snst
    rxRD_txL_peakAmps = shared_pwr_txL_to_rxR[smp_lo:smp_hi,3]*pd_snst
    rxLA_txR_peakAmps = shared_pwr_txR_to_rxL[smp_lo:smp_hi,0]*pd_snst
    rxLC_txR_peakAmps = shared_pwr_txR_to_rxL[smp_lo:smp_hi,1]*pd_snst
    rxLB_txR_peakAmps = shared_pwr_txR_to_rxL[smp_lo:smp_hi,2]*pd_snst
    rxLD_txR_peakAmps = shared_pwr_txR_to_rxL[smp_lo:smp_hi,3]*pd_snst
    rxRA_txR_peakAmps = shared_pwr_txR_to_rxR[smp_lo:smp_hi,0]*pd_snst
    rxRC_txR_peakAmps = shared_pwr_txR_to_rxR[smp_lo:smp_hi,1]*pd_snst
    rxRB_txR_peakAmps = shared_pwr_txR_to_rxR[smp_lo:smp_hi,2]*pd_snst
    rxRD_txR_peakAmps = shared_pwr_txR_to_rxR[smp_lo:smp_hi,3]*pd_snst

    dLL_sigTime  = np.interp(simulation_time, step_time, delay_txL_to_rxL[smp_lo:smp_hi])
    dLR_sigTime  = np.interp(simulation_time, step_time, delay_txL_to_rxR[smp_lo:smp_hi])
    dRL_sigTime  = np.interp(simulation_time, step_time, delay_txR_to_rxL[smp_lo:smp_hi])
    dRR_sigTime  = np.interp(simulation_time, step_time, delay_txR_to_rxR[smp_lo:smp_hi])

    ### /2 because sin is -1 to +1, which is 2 A_pp, we want nominal 1 A_pp since we're mapping to 
    ### full intensity of the TX beam, and the pre-amp at the TIA is AC-coupled 
    rxLA_txL_peakAmps_sigTime = np.interp(simulation_time, step_time, rxLA_txL_peakAmps) 
    rxLA_txL_wavAmps          = rxLA_txL_peakAmps_sigTime*(np.sin(2*np.pi*f_eL*(simulation_time - dLL_sigTime) )) / 2; 
    rxLB_txL_peakAmps_sigTime = np.interp(simulation_time, step_time, rxLB_txL_peakAmps) 
    rxLB_txL_wavAmps          = rxLB_txL_peakAmps_sigTime*(np.sin(2*np.pi*f_eL*(simulation_time - dLL_sigTime) )) / 2; 
    rxLC_txL_peakAmps_sigTime = np.interp(simulation_time, step_time, rxLC_txL_peakAmps) 
    rxLC_txL_wavAmps          = rxLC_txL_peakAmps_sigTime*(np.sin(2*np.pi*f_eL*(simulation_time - dLL_sigTime) )) / 2; 
    rxLD_txL_peakAmps_sigTime = np.interp(simulation_time, step_time, rxLD_txL_peakAmps) 
    rxLD_txL_wavAmps          = rxLD_txL_peakAmps_sigTime*(np.sin(2*np.pi*f_eL*(simulation_time - dLL_sigTime) )) / 2; 

    rxRA_txL_peakAmps_sigTime = np.interp(simulation_time, step_time, rxRA_txL_peakAmps) 
    rxRA_txL_wavAmps          = rxRA_txL_peakAmps_sigTime*(np.sin(2*np.pi*f_eL*(simulation_time - dLR_sigTime) )) / 2; 
    rxRB_txL_peakAmps_sigTime = np.interp(simulation_time, step_time, rxRB_txL_peakAmps) 
    rxRB_txL_wavAmps          = rxRB_txL_peakAmps_sigTime*(np.sin(2*np.pi*f_eL*(simulation_time - dLR_sigTime) )) / 2; 
    rxRC_txL_peakAmps_sigTime = np.interp(simulation_time, step_time, rxRC_txL_peakAmps) 
    rxRC_txL_wavAmps          = rxRC_txL_peakAmps_sigTime*(np.sin(2*np.pi*f_eL*(simulation_time - dLR_sigTime) )) / 2; 
    rxRD_txL_peakAmps_sigTime = np.interp(simulation_time, step_time, rxRD_txL_peakAmps) 
    rxRD_txL_wavAmps          = rxRD_txL_peakAmps_sigTime*(np.sin(2*np.pi*f_eL*(simulation_time - dLR_sigTime) )) / 2; 

    rxLA_txR_peakAmps_sigTime = np.interp(simulation_time, step_time, rxLA_txR_peakAmps) 
    rxLA_txR_wavAmps          = rxLA_txR_peakAmps_sigTime*(np.sin(2*np.pi*f_eR*(simulation_time - dRL_sigTime) )) / 2; 
    rxLB_txR_peakAmps_sigTime = np.interp(simulation_time, step_time, rxLB_txR_peakAmps) 
    rxLB_txR_wavAmps          = rxLB_txR_peakAmps_sigTime*(np.sin(2*np.pi*f_eR*(simulation_time - dRL_sigTime) )) / 2; 
    rxLC_txR_peakAmps_sigTime = np.interp(simulation_time, step_time, rxLC_txR_peakAmps) 
    rxLC_txR_wavAmps          = rxLC_txR_peakAmps_sigTime*(np.sin(2*np.pi*f_eR*(simulation_time - dRL_sigTime) )) / 2; 
    rxLD_txR_peakAmps_sigTime = np.interp(simulation_time, step_time, rxLD_txR_peakAmps) 
    rxLD_txR_wavAmps          = rxLD_txR_peakAmps_sigTime*(np.sin(2*np.pi*f_eR*(simulation_time - dRL_sigTime) )) / 2; 

    rxRA_txR_peakAmps_sigTime = np.interp(simulation_time, step_time, rxRA_txR_peakAmps) 
    rxRA_txR_wavAmps          = rxRA_txR_peakAmps_sigTime*(np.sin(2*np.pi*f_eR*(simulation_time - dRR_sigTime) )) / 2; 
    rxRB_txR_peakAmps_sigTime = np.interp(simulation_time, step_time, rxRB_txR_peakAmps) 
    rxRB_txR_wavAmps          = rxRB_txR_peakAmps_sigTime*(np.sin(2*np.pi*f_eR*(simulation_time - dRR_sigTime) )) / 2; 
    rxRC_txR_peakAmps_sigTime = np.interp(simulation_time, step_time, rxRC_txR_peakAmps) 
    rxRC_txR_wavAmps          = rxRC_txR_peakAmps_sigTime*(np.sin(2*np.pi*f_eR*(simulation_time - dRR_sigTime) )) / 2; 
    rxRD_txR_peakAmps_sigTime = np.interp(simulation_time, step_time, rxRD_txR_peakAmps) 
    rxRD_txR_wavAmps          = rxRD_txR_peakAmps_sigTime*(np.sin(2*np.pi*f_eR*(simulation_time - dRR_sigTime) )) / 2; 

    rxLA_total_pwr = shared_pwr_txR_to_rxL[smp_lo:smp_hi,0] + shared_pwr_txL_to_rxL[smp_lo:smp_hi,0] 
    rxLB_total_pwr = shared_pwr_txR_to_rxL[smp_lo:smp_hi,1] + shared_pwr_txL_to_rxL[smp_lo:smp_hi,1] 
    rxLC_total_pwr = shared_pwr_txR_to_rxL[smp_lo:smp_hi,2] + shared_pwr_txL_to_rxL[smp_lo:smp_hi,2] 
    rxLD_total_pwr = shared_pwr_txR_to_rxL[smp_lo:smp_hi,3] + shared_pwr_txL_to_rxL[smp_lo:smp_hi,3]
    rxRA_total_pwr = shared_pwr_txR_to_rxR[smp_lo:smp_hi,0] + shared_pwr_txL_to_rxR[smp_lo:smp_hi,0] 
    rxRB_total_pwr = shared_pwr_txR_to_rxR[smp_lo:smp_hi,1] + shared_pwr_txL_to_rxR[smp_lo:smp_hi,1] 
    rxRC_total_pwr = shared_pwr_txR_to_rxR[smp_lo:smp_hi,2] + shared_pwr_txL_to_rxR[smp_lo:smp_hi,2] 
    rxRD_total_pwr = shared_pwr_txR_to_rxR[smp_lo:smp_hi,3] + shared_pwr_txL_to_rxR[smp_lo:smp_hi,3]

    rxLA_noise_var = rx_P_rx_factor*rxLA_total_pwr + thermal_and_bg_curr; 
    rxLB_noise_var = rx_P_rx_factor*rxLB_total_pwr + thermal_and_bg_curr; 
    rxLC_noise_var = rx_P_rx_factor*rxLC_total_pwr + thermal_and_bg_curr; 
    rxLD_noise_var = rx_P_rx_factor*rxLD_total_pwr + thermal_and_bg_curr; 
    rxRA_noise_var = rx_P_rx_factor*rxRA_total_pwr + thermal_and_bg_curr; 
    rxRB_noise_var = rx_P_rx_factor*rxRB_total_pwr + thermal_and_bg_curr; 
    rxRC_noise_var = rx_P_rx_factor*rxRC_total_pwr + thermal_and_bg_curr; 
    rxRD_noise_var = rx_P_rx_factor*rxRD_total_pwr + thermal_and_bg_curr; 

    ### add noise, get received signals at each quadrant in sig_time
    ### note that the signals are separate for the two TX units -> this is 
    ### because we assume that the two TX signals are kept at different frequency bands, 
    ### thus, they can be easily extracted from the actual received signal via bandpass filtering. 
    numsamples = len(rxLA_txL_wavAmps)

    rxLA_noise_std = np.interp(simulation_time, step_time, np.sqrt(rxLA_noise_var));
    rxLB_noise_std = np.interp(simulation_time, step_time, np.sqrt(rxLB_noise_var));
    rxLC_noise_std = np.interp(simulation_time, step_time, np.sqrt(rxLC_noise_var));
    rxLD_noise_std = np.interp(simulation_time, step_time, np.sqrt(rxLD_noise_var));

    rxLA_txL = (rxLA_txL_wavAmps + add_noise * rxLA_noise_std * np.random.randn(numsamples))*pd_gain;
    rxLB_txL = (rxLB_txL_wavAmps + add_noise * rxLB_noise_std * np.random.randn(numsamples))*pd_gain;
    rxLC_txL = (rxLC_txL_wavAmps + add_noise * rxLC_noise_std * np.random.randn(numsamples))*pd_gain;
    rxLD_txL = (rxLD_txL_wavAmps + add_noise * rxLD_noise_std * np.random.randn(numsamples))*pd_gain;
    rxLA_txR = (rxLA_txR_wavAmps + add_noise * rxLA_noise_std * np.random.randn(numsamples))*pd_gain;
    rxLB_txR = (rxLB_txR_wavAmps + add_noise * rxLB_noise_std * np.random.randn(numsamples))*pd_gain;
    rxLC_txR = (rxLC_txR_wavAmps + add_noise * rxLC_noise_std * np.random.randn(numsamples))*pd_gain;
    rxLD_txR = (rxLD_txR_wavAmps + add_noise * rxLD_noise_std * np.random.randn(numsamples))*pd_gain;

    rxRA_noise_std = np.interp(simulation_time, step_time, np.sqrt(rxRA_noise_var));
    rxRB_noise_std = np.interp(simulation_time, step_time, np.sqrt(rxRB_noise_var));
    rxRC_noise_std = np.interp(simulation_time, step_time, np.sqrt(rxRC_noise_var));
    rxRD_noise_std = np.interp(simulation_time, step_time, np.sqrt(rxRD_noise_var));

    rxRA_txL = (rxRA_txL_wavAmps + add_noise * rxRA_noise_std * np.random.randn(numsamples))*pd_gain;
    rxRB_txL = (rxRB_txL_wavAmps + add_noise * rxRB_noise_std * np.random.randn(numsamples))*pd_gain;
    rxRC_txL = (rxRC_txL_wavAmps + add_noise * rxRC_noise_std * np.random.randn(numsamples))*pd_gain;
    rxRD_txL = (rxRD_txL_wavAmps + add_noise * rxRD_noise_std * np.random.randn(numsamples))*pd_gain;
    rxRA_txR = (rxRA_txR_wavAmps + add_noise * rxRA_noise_std * np.random.randn(numsamples))*pd_gain;
    rxRB_txR = (rxRB_txR_wavAmps + add_noise * rxRB_noise_std * np.random.randn(numsamples))*pd_gain;
    rxRC_txR = (rxRC_txR_wavAmps + add_noise * rxRC_noise_std * np.random.randn(numsamples))*pd_gain;
    rxRD_txR = (rxRD_txR_wavAmps + add_noise * rxRD_noise_std * np.random.randn(numsamples))*pd_gain;

    return (rxLA_txL, rxLB_txL, rxLC_txL, rxLD_txL), (rxLA_txR, rxLB_txR, rxLC_txR, rxLD_txR), (rxRA_txL, rxRB_txL, rxRC_txL, rxRD_txL), (rxRA_txR, rxRB_txR, rxRC_txR, rxRD_txR), (dLL_sigTime, dLR_sigTime, dRL_sigTime, dRR_sigTime)

##############################################################################
### utility functions

def map_rx_config(input_dict):
    f_QRX    = input_dict['f_QRX']
    pd_snst  = input_dict['tia_gamma']
    pd_gain  = input_dict['tia_R_F']
    pd_dim   = input_dict['detecting_area'] # square detector dimension in milimeters. needs a /1000 in calcs
    rx_P_rx_factor     = input_dict['tia_shot_P_r_factor']
    rx_I_bg_factor     = input_dict['tia_shot_I_bg_factor']
    rx_thermal_factor1 = input_dict['tia_thermal_factor1']
    rx_thermal_factor2 = input_dict['tia_thermal_factor2']

    return f_QRX, pd_snst, pd_gain, pd_dim, rx_P_rx_factor, rx_I_bg_factor, rx_thermal_factor1, rx_thermal_factor2

def map_tx_config(input_dict):
    tx_ha   = input_dict['half_angle']
    tx_pwr  = input_dict['power']
    tx_norm = input_dict['normalization_factor']
    tx_lambertian_order = int(-np.log(2)/np.log(np.cos(np.deg2rad(tx_ha))));

    return tx_ha, tx_pwr, tx_norm, tx_lambertian_order

def map_tx_config_nonlambertian(input_dict):
    tx_pattern    = input_dict['pattern']
    tx_phiarray   = input_dict['phi_array']
    tx_thetaarray = input_dict['theta_array']
    tx_pwr        = input_dict['power']
    tx_norm       = input_dict['normalization_factor']
    
    return tx_pattern, tx_phiarray, tx_thetaarray, tx_pwr, tx_norm
