import numpy as np

##############################################################################
### Method 1 - Approximate the 3D Lambertian pattern with a rectangular prism  
###            for very small angular intervals (fast, but biased/approximate)
def tx_solidangles(x, y, z, dim):
    # note that axis xz is not used since roads are assumed to be flat
    psi1_xy = np.arctan2(y,x + dim/2);
    psi2_xy = np.arctan2(y,x - dim/2);
    psi3_xy = psi2_xy - psi1_xy;
    eps1_xy = np.rad2deg((np.deg2rad(90)-psi2_xy));
    eps2_xy = np.rad2deg(np.deg2rad(eps1_xy) + psi3_xy);

    psi1_zy = np.arctan2(y,z + dim/2);
    psi2_zy = np.arctan2(y,z - dim/2);
    psi3_zy = psi2_zy - psi1_zy;
    eps1_zy = np.rad2deg((np.deg2rad(90)-psi2_zy));
    eps2_zy = np.rad2deg(np.deg2rad(eps1_zy) + psi3_zy);
    
    return eps1_xy, eps2_xy, eps1_zy, eps2_zy

def received_power(x, y, z, dim, tx_pwr, tx_norm, tx_lambertian_order, attenuation_factor):
    # the portion of the lambertian pattern staying within these angles is a rectangular prism with a "wavy" top. 
    # prism is smooth though, + we always check veeery small solidangle portions, so we can assume
    # the top is close to being a linear slump that is so smooth, it's nearly a rectangular prism
    # with height = average height of the four corners. This should bring very small error. 
    angles = tx_solidangles(x, y, z, dim)
    #angles = (eps1_xy, eps2_xy, eps1_zy, eps2_zy)
        
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

##############################################################################
### Method 2 - Numerical integration (very slow, but unbiased) 

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
### Quad detector function fit

def find_closest_linear_fit(fqrx_samples, actual_angle):
    fqrx_samples_rep = np.expand_dims(fqrx_samples, axis=0)
    fqrx_samples_rep = np.repeat(fqrx_samples_rep, actual_angle.shape[0], axis=0)
    actual_angle_exp = np.expand_dims(actual_angle, axis=1)

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

    A_sh = A_sh/2;
    B_sh = B_sh/2;
    C_sh = C_sh/2;
    D_sh = D_sh/2;
    return A_sh, B_sh, C_sh, D_sh
    
def quad_distribute_power(x,y,z,fqrx, power):
    A_sh, B_sh, C_sh, D_sh = quad_shares_power(x,y,z,fqrx);
    A_pwr = power*A_sh;
    B_pwr = power*B_sh;
    C_pwr = power*C_sh;
    D_pwr = power*D_sh;
    return A_pwr, B_pwr, C_pwr, D_pwr

##############################################################################
### propagation delay

def propagation_delay(x,y,z):
    # we ignore z
    speed_of_light    = 299792458 # [m/s]
    delay = np.sqrt(x**2 + y**2)/speed_of_light;
    return delay
