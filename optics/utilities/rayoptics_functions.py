from rayoptics.environment import *
from rayoptics.raytr import sampler
import numpy as np
import os


## initializations specific to _ray-optics_ for a given setup. 
## note that this implies a fixed (very simple) optical setup, 
## it's the one discussed in: https://github.com/mjhoptics/ray-optics/discussions/45
def define_optics(lens_codv_name, detector_plane, aperture_stop, aoa_list):
    opm = open_model(lens_codv_name)
    sm  = opm.seq_model
    osp = opm.optical_spec
    pm = opm.parax_model

    sm.gaps[2].thi = detector_plane
    osp.pupil = PupilSpec(osp, key=['object', 'pupil'], value = aperture_stop)
    sm.set_cur_surface(2) 
    sm.set_stop()
    
    osp.field_of_view = FieldSpec(osp, flds = aoa_list)

    opm.update_model()
    
    return opm, sm, osp, pm


## tracing rays through the optical setup 
def get_ray_lists(num_rays_sqrt, limits, osp, opm, draw_spots=False):
    num_flds = len(osp.field_of_view.fields)

    # trace rays at each field and save results
    num_rays= num_rays_sqrt
    us = limits
    ray_lists = []
    on_axis_pt = np.array([0.0, 0.0])
    for f in range(num_flds):
        fld, wvl, foc = osp.lookup_fld_wvl_focus(f)

        r2g      = (sampler.create_generator, (sampler.R_2_quasi_random_generator, num_rays**2), dict(mapper=sampler.concentric_sample_disk))
        ray_lists += [analyses.RayList(opm, pupil_gen=r2g, f=fld, wl=wvl, image_pt_2d=on_axis_pt)]

    if(draw_spots):
        subplots = [RayGeoPSF(ray_list, user_scale_value=us, scale_type='user', dsp_typ='hist2d',
                              yaxis_ticks_position='right', cmap='gray',
                              title=f'{osp.field_of_view.fields[f].y} deg') for f, ray_list in enumerate(ray_lists)]

        # set up a figure and draw the irradiance maps into it
        irrdfig = plt.figure(FigureClass=AnalysisFigure, data_objs=ray_lists, subplots=subplots, grid=(1, num_flds),
                             figsize=[20, 10], dpi=150, tight_layout=True, is_dark=True).plot()
    
    return ray_lists

## compute transmission coefficients through surfaces (used in fresnel loss computations)
def calc_transmission_coefs(ray_list, rndx, idx):
    raydirvec_front        = ray_list[idx][2][0][0][1];
    raydirvec_back         = ray_list[idx][2][0][1][1];
    raydirvec_out          = ray_list[idx][2][0][2][1];
    surfnormalvec_front    = ray_list[idx][2][0][1][3];
    surfnormalvec_back     = ray_list[idx][2][0][2][3];

    # first surface, lens front (curved) 
    n1 = rndx[0][0]; #air
    n2 = rndx[1][0]; #lens
    # use dot product for cosine. direction and normal vectors are already normalized
    cos_ai = np.dot(raydirvec_front, surfnormalvec_front)
    cos_at = np.dot(raydirvec_back, surfnormalvec_front)
    R_s_polarized        = ((n1*cos_ai - n2*cos_at)/(n1*cos_ai + n2*cos_at))**2;
    R_p_polarized        = ((n1*cos_at - n2*cos_ai)/(n1*cos_at + n2*cos_ai))**2;
    R_effective          = (R_s_polarized + R_p_polarized)/2
    transmit_coef_front  = 1-R_effective

    # second surface, lens back (straight)
    n1 = rndx[1][0]; #lens
    n2 = rndx[2][0]; #air
    # use dot product for cosine. direction and normal vectors are already normalized
    cos_ai = np.dot(raydirvec_back, surfnormalvec_back)
    cos_at = np.dot(raydirvec_out, surfnormalvec_back)
    R_s_polarized        = ((n1*cos_ai - n2*cos_at)/(n1*cos_ai + n2*cos_at))**2;
    R_p_polarized        = ((n1*cos_at - n2*cos_ai)/(n1*cos_at + n2*cos_ai))**2;
    R_effective          = (R_s_polarized + R_p_polarized)/2
    transmit_coef_back   = 1-R_effective
    
    return [transmit_coef_front, transmit_coef_back]

## compute angular response for the setup discussed in:
## https://github.com/mjhoptics/ray-optics/discussions/45
## i.e., a quad detector and a lens+aperture stop right above it
## note that there is an option for taking the fresnel losses into account
def calc_angular_response(ray_list, rndx, with_fresnel=False):
    xy_of_rays_on_spot = np.zeros((1,2)) # [0]: x, [1]: y
    
    # this "index train" eventually gives: 
    #      xy position ( via [0][0:2]) 
    #      on the 4th surface ( via [3], due to 0-start-indexing )  
    #      for ray idx 0 ( via the first [0])
    #      and the [2][0] in the middle does not have physical meaning
    xy_of_rays_on_spot[0]  = ray_list[0][2][0][3][0][0:2];     
    for ray_idx in range(1,len(ray_list)):
        xy_of_rays_on_spot = np.concatenate((xy_of_rays_on_spot, 
                                             np.expand_dims(ray_list[ray_idx][2][0][3][0][0:2],axis=0)))
    x = xy_of_rays_on_spot[:,0];
    y = xy_of_rays_on_spot[:,1];
    
    # assuming a quadrant detector, size not limited
    qA_rays     = np.logical_and((x > 0),(y > 0))
    qB_rays     = np.logical_and((x < 0),(y > 0))
    qC_rays     = np.logical_and((x > 0),(y < 0))
    qD_rays     = np.logical_and((x < 0),(y < 0))
    qA_num_rays = np.count_nonzero( qA_rays )
    qB_num_rays = np.count_nonzero( qB_rays )
    qC_num_rays = np.count_nonzero( qC_rays )
    qD_num_rays = np.count_nonzero( qD_rays )

    # phi denotes the "power ratio" among quadrants
    # this is without fresnel losses
    total  = qA_num_rays + qB_num_rays + qC_num_rays + qD_num_rays
    phi_AC = ( (qA_num_rays + qC_num_rays) - (qB_num_rays + qD_num_rays) ) / total ;
    phi_AB = ( (qA_num_rays + qB_num_rays) - (qC_num_rays + qD_num_rays) ) / total ;

    # handle fresnel losses here
    phi_AB_wf = None
    phi_AC_wf = None
    if(with_fresnel):
        coefs_on_surfaces    = np.zeros((1,2)) # [0]: from lens front curved, [1]: from lens back straight 
        coefs_on_surfaces[0] = calc_transmission_coefs(ray_list, rndx, 0);
        
        for ray_idx in range(1,len(ray_list)):
            tt                = calc_transmission_coefs(ray_list, rndx, ray_idx);
            coefs_on_surfaces = np.concatenate((coefs_on_surfaces, np.expand_dims(tt,axis=0)))

        transmission_scaler = coefs_on_surfaces[:,0]*coefs_on_surfaces[:,1];
        qA_rays_weighted = np.sum(qA_rays.astype(np.float)*transmission_scaler)
        qB_rays_weighted = np.sum(qB_rays.astype(np.float)*transmission_scaler)
        qC_rays_weighted = np.sum(qC_rays.astype(np.float)*transmission_scaler)
        qD_rays_weighted = np.sum(qD_rays.astype(np.float)*transmission_scaler)
        
        # same as above, but this time with fresnel losses
        total     = qA_rays_weighted + qB_rays_weighted + qC_rays_weighted + qD_rays_weighted
        phi_AC_wf = ( (qA_rays_weighted + qC_rays_weighted) - (qB_rays_weighted + qD_rays_weighted) ) / total ;
        phi_AB_wf = ( (qA_rays_weighted + qB_rays_weighted) - (qC_rays_weighted + qD_rays_weighted) ) / total ;

    return phi_AB, phi_AC, phi_AB_wf, phi_AC_wf, total

## just a wrapper for the fcn above
def calc_fqrx(osp, raylist, aoa_list, rndx, draw_fqrx=False, with_fresnel=False):
    num_flds = len(osp.field_of_view.fields)
    phi_AB_list    = []
    phi_AB_wf_list = []
    total_list = []
    for i in range(0, num_flds):
        ray_list_for_aoa = raylist[i].ray_list;
        
        # just compute the horizontal response, that's enough, system is radially symmetric, so drop ACs
        phi_AB, _, phi_AB_wf, _, total = calc_angular_response(ray_list_for_aoa, rndx, with_fresnel=with_fresnel)
        phi_AB_list.append(phi_AB)
        if(with_fresnel):
            phi_AB_wf_list.append(phi_AB_wf)
        total_list.append(total)

    if(draw_fqrx):
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(aoa_list, phi_AB_list)
        if(with_fresnel):
            ax.plot(aoa_list, phi_AB_wf_list)
            ax.legend(['without Fresnel losses', 'with Fresnel losses'])
        ax.set_xlim([-90, 90])
        ax.set_ylim([-1.1, 1.1])
        plt.show()
    return phi_AB_list, phi_AB_wf_list, total_list
