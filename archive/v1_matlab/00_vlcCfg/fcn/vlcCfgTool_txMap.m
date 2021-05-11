function [ map ] = vlcCfgTool_txMap( optical, mode , fit_resolution_numOfPts)
%VLCCFGTOOL_FOVLUT Summary of this function goes here
%   Detailed explanation goes here
addpath('fcn');

f = waitbar(0, 'Prepaing normalized TX beam pattern');
if(strcmp(mode,'lambertian_radSym')) % Lambertian, radially symmetric

    %%% NOTE: this is not actually a lambertian since we're approximating
    %%%       the pattern with a smoothspline interpolator with
    %%%       fit_resolution_numOfPts below
    
    angle_array_square = (-90*sqrt(2)-1):optical.simRes_theta:(90*sqrt(2)+1); % enlarge square array by a bit so that the integrator doesn't hit bounds
    lambertian_order   = round(-log(2)/log(cosd(optical.half_angle)));
    pattern            = cosd(angle_array_square).^lambertian_order;
    pattern_indexed    = [angle_array_square;pattern];
    vol                = vlcCfgTool_radSymSrc3dIntegral( pattern_indexed, -90, 90, -90, 90);   % will be used for normalization later
    clear lambertian_order pattern angle_array_square
elseif(strcmp(mode,'custom_radSym')) % custom, radially symmetric
    % old implementation here
    fn = optical.num_pattern_fn;    load(fn);
    vol = vlcCfgTool_radSymSrc3dIntegral( vlcCfgTxPlrPatNum, -90, 90, -90, 90);
    pattern_indexed = vlcCfgTxPlrPatNum;
    clear vlcCfgTxPlrPatNum
elseif(mode=='custom_notSym') % totally custom, not necessarily radially symmetric
    disp('Full custom mode not implemented yet');
    return
    % VEINS here
else
    disp('wrong mode choice');
    return 
    %?!!?
end
waitbar(0.25,f, 'Preparing normalized TX beam pattern');

subsampling_factor = round(size(pattern_indexed,2)/fit_resolution_numOfPts);
% Define the initial profile
x = pattern_indexed(1,1:subsampling_factor:end);
a = abs(x-(-90));  m90_pos = find(a == min(a));
a = abs(x-(90));   p90_pos = find(a == min(a));
y = pattern_indexed(2,1:subsampling_factor:end);

% Compute normalized 3D pattern (it's a 2D matrix with "height" values for the surface)
pat = zeros(length(y));
for i = m90_pos:p90_pos
    for j = m90_pos:p90_pos
        d = abs(x-sqrt(x(i)*x(i)+x(j)*x(j)));
        a = find(d == min(d));
        pat(i,j) = y(a)/vol;
    end
end
waitbar(0.50,f, 'Preparing normalized TX beam pattern');

% Surface fitting
clearvars -except f pat x y pattern_indexed
x_long = repmat(x,[1 length(x)]);
y_long = zeros(size(x_long));
for i=1:length(x_long)
    y_long(i) = x_long(ceil(i/sqrt(length(x_long))));
end
pat_long = reshape(pat,[1 size(pat,1)*size(pat,2)]);
waitbar(0.75,f, 'Preparing normalized TX beam pattern');
map = fit([x_long',y_long'],pat_long','cubicinterp');

% Test the fit, tic toc to understand how much it's going to take to
% calculate a 2 deg (hrz and vert) rectangular area during the actual
% simulation. May want to increase the subsampling rate to make it faster
figure
plot(map,[x_long',y_long'],pat_long'), title('Normalized TX Beam Pattern')
disp(sprintf('Area under curve for -1 to 1 degree square: %f', quad2d(map,-1,1,-1,1)));
close(f)
end

