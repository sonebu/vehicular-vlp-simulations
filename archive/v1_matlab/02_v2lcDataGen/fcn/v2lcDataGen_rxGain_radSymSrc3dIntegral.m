function [ vol ] = v2lcDataGen_rxGain_radSymSrc3dIntegral( rad_pat, eps1_xy, eps2_xy, eps1_zy, eps2_zy )
%v2lcDataGen_rxGain_radSymSrc3dIntegral Summary of this function goes here
%   Detailed explanation goes here

%%% This is our custom integrator, but wasn't used, kept here for
%%% completeness. We used to use this for accurately computing the AUC of
%%% the TX pattern for normalizing it. This made sense during the Erdem's simit
%%% approximation and the rectangular approximation, but we dropped those
%%% now and focused on Lambertian approximations of radially symmetric
%%% patterns due to the fact that MAJORITY of the existing works use that.
%%% We'll handle ECE-compliant patterns in future work.

id0_eps1_xy = find(abs(rad_pat(1,:)-eps1_xy)==min(abs(rad_pat(1,:)-eps1_xy)));
id0_eps2_xy = find(abs(rad_pat(1,:)-eps2_xy)==min(abs(rad_pat(1,:)-eps2_xy)));
id0_eps1_zy = find(abs(rad_pat(1,:)-eps1_zy)==min(abs(rad_pat(1,:)-eps1_zy)));
id0_eps2_zy = find(abs(rad_pat(1,:)-eps2_zy)==min(abs(rad_pat(1,:)-eps2_zy)));
id_zero = round(size(rad_pat,2)/2);
id_list_fwd_xy = linspace(id0_eps2_xy,id0_eps1_xy,abs(id0_eps2_xy-id0_eps1_xy)+1);
id_list_bwd_xy = linspace(id0_eps1_xy,id0_eps2_xy,abs(id0_eps2_xy-id0_eps1_xy)+1);
id_list_fwd_zy = linspace(id0_eps2_zy,id0_eps1_zy,abs(id0_eps2_zy-id0_eps1_zy)+1);
id_list_bwd_zy = linspace(id0_eps1_zy,id0_eps2_zy,abs(id0_eps2_zy-id0_eps1_zy)+1);
vol_fwd = 0;
for i=2:length(id_list_fwd_xy)
    for j=2:length(id_list_fwd_zy)
        radial_pos_i_xy_i_zy     = round(sqrt((id_list_fwd_xy(i)-id_zero)^2   + (id_list_fwd_zy(j)-id_zero)^2));
        %         radial_pos_im1_xy_i_zy   = round(sqrt((id_list_fwd_xy(i-1)-id_zero)^2 + (id_list_fwd_zy(j)-id_zero)^2));
        %         radial_pos_i_xy_im1_zy   = round(sqrt((id_list_fwd_xy(1)-id_zero)^2   + (id_list_fwd_zy(j-1)-id_zero)^2));
        %         radial_pos_im1_xy_im1_zy = round(sqrt((id_list_fwd_xy(i-1)-id_zero)^2 + (id_list_fwd_zy(j-1)-id_zero)^2));
        radial_pos_im1_xy_i_zy   = radial_pos_i_xy_i_zy-1;
        radial_pos_i_xy_im1_zy   = radial_pos_i_xy_i_zy-1;
        radial_pos_im1_xy_im1_zy = radial_pos_i_xy_i_zy-2;
        avg_y_val = (rad_pat(2,radial_pos_i_xy_i_zy+id_zero) + ...
            rad_pat(2,radial_pos_im1_xy_i_zy+id_zero) + ...
            rad_pat(2,radial_pos_i_xy_im1_zy+id_zero) + ...
            rad_pat(2,radial_pos_im1_xy_im1_zy+id_zero) )/4 ;
        vol_fwd = vol_fwd + (rad_pat(1,radial_pos_i_xy_i_zy+id_zero)-rad_pat(1,radial_pos_im1_xy_i_zy+id_zero))*...
            (rad_pat(1,radial_pos_i_xy_i_zy+id_zero)-rad_pat(1,radial_pos_i_xy_im1_zy+id_zero))*...
            avg_y_val;
    end
end

vol_bwd = 0;
for i=2:length(id_list_bwd_xy)
    for j=2:length(id_list_bwd_zy)
        radial_pos_i_xy_i_zy     = round(sqrt((id_list_bwd_xy(i)-id_zero)^2   + (id_list_bwd_zy(j)-id_zero)^2));
        %         radial_pos_im1_xy_i_zy   = round(sqrt((id_list_bwd_xy(i-1)-id_zero)^2 + (id_list_bwd_zy(j)-id_zero)^2));
        %         radial_pos_i_xy_im1_zy   = round(sqrt((id_list_bwd_xy(1)-id_zero)^2   + (id_list_bwd_zy(j-1)-id_zero)^2));
        %         radial_pos_im1_xy_im1_zy = round(sqrt((id_list_bwd_xy(i-1)-id_zero)^2 + (id_list_bwd_zy(j-1)-id_zero)^2));
        radial_pos_im1_xy_i_zy   = radial_pos_i_xy_i_zy+1;
        radial_pos_i_xy_im1_zy   = radial_pos_i_xy_i_zy+1;
        radial_pos_im1_xy_im1_zy = radial_pos_i_xy_i_zy+2;
        avg_y_val = (rad_pat(2,radial_pos_i_xy_i_zy+id_zero) + ...
            rad_pat(2,radial_pos_im1_xy_i_zy+id_zero) + ...
            rad_pat(2,radial_pos_i_xy_im1_zy+id_zero) + ...
            rad_pat(2,radial_pos_im1_xy_im1_zy+id_zero) )/4 ;
        vol_bwd = vol_bwd + (rad_pat(1,radial_pos_i_xy_i_zy+id_zero)-rad_pat(1,radial_pos_im1_xy_i_zy+id_zero))*...
            (rad_pat(1,radial_pos_i_xy_i_zy+id_zero)-rad_pat(1,radial_pos_i_xy_im1_zy+id_zero))*...
            avg_y_val;
    end
end
vol = (vol_fwd+vol_bwd)/2;

end

