function [ rx_gain] = v2lcDataGen_rxGain( txPlrPat, x, y, z, pd_ddl)
%v2lcDataGen_rxGain Summary of this function goes here
%   Detailed explanation goes here

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% (OBSOLETE) WE'VE MADE AN APPROXIMATION HERE                        %%%
%%%                                                                    %%%
%%% At the angular resolution we want (max 0.0001 deg), MATLAB can't   %%%
%%% store the 3d surface representing the radiation pattern. So volume %%%
%%% under that curve was approximated by Erdem's simit method.         %%%
%%% Compute AUC for xy, revolve it 360 deg. repeat for zy              %%%
%%% You get two rings, extract the part from xy which spans angle by zy%%%
%%% repeat for zy -> xy get the average                                %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% (OBSOLETE) UPDATE: There's also a rectangle approximation actually...   %%%
%%% Since our source is radially symmetric, the tilt and azimuth angles are %%%
%%% affixed to the base of the 3d plot, this defines a rectangle. We divide %%%
%%% this into as many parts as the resolution we desire (maybe we can think %%%
%%% of efficiency gains with downsampling in the future). The values these  %%%
%%% parts take in the perpendicular axis of the 3d plot (vertical) lets     %%%
%%% call them phi and ksi, sqrt(phi^2 + ksi^2) is our original rad_pat val  %%%
%%% THIS ONLY WORKS BECAUSE THE SOURCE IS RADIALLY SYMMETRIC!!              %%%
%%% Non-radially symmetric sources can maybe be represented as sum of       %%%
%%% radially symmetric sources but that should be automatized and currently %%%
%%% is not implemented. This would actually be good because automotive beam %%%
%%% patterns are pretty irregular                                           %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% UPDATE: OK dropped irregular polar patterns for now. We'll leave        %%%
%%% ECE-compliant patterns as future work. With this, since Lambertian      %%%
%%% approximations are cosine power functions (veeery smooth!), we'll just  %%%
%%% sub-sample, fit, and interpolate with bicubic. The Lambertian is        %%%
%%% unrealistic enough itself but becha uses it, so no need to kill         %%%
%%% ourselves over the accuracy of the Lambertian pattern...                %%%
%%% summary: just fit subsampled pattern and use quad2d, that's efficient   %%%
%%% enough                                                                  %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Geometric calc on XY
psi1_xy = atan2(y,x + pd_ddl/2);
psi2_xy = atan2(y,x - pd_ddl/2);
psi3_xy = psi2_xy - psi1_xy;
% eps1_xy = rad2deg((deg2rad(90)-psi2_xy)-hdg); % this axis is not used, flat roads
eps1_xy = rad2deg((deg2rad(90)-psi2_xy));
eps2_xy = rad2deg(deg2rad(eps1_xy) + psi3_xy);

% Geometric calc on ZY
psi1_zy = atan2(y,z + pd_ddl/2);
psi2_zy = atan2(y,z - pd_ddl/2);
psi3_zy = psi2_zy - psi1_zy;
% eps1_zy = rad2deg((deg2rad(90)-psi2_zy)-hdg); % this axis is not used, flat roads
eps1_zy = rad2deg((deg2rad(90)-psi2_zy));
eps2_zy = rad2deg(deg2rad(eps1_zy) + psi3_zy);

vol = quad2d(txPlrPat,eps1_xy,eps2_xy,eps1_zy,eps2_zy);
rx_gain = vol;

end

