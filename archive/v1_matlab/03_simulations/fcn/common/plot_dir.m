function [h1, h2] = plot_dir (vX, vY, relative_heading, subsampling)
%function [h1, h2] = plot_dir (vX, vY)
%Plotting x-y variables with direction indicating vector to the next element.
%Example
%   vX = linspace(0,2*pi, 10)';
%   vY = sin (vX);
%   plot_dir(vX, vY);

rMag = 1;

% Length of vector
lenTime = length(vX);

% Indices of tails of arrows
vSelect0 = 1:subsampling:(lenTime-1);

% X coordinates of tails of arrows
vXQ0 = vX(vSelect0, 1);
% Y coordinates of tails of arrows
vYQ0 = vY(vSelect0, 1);

% X coordinates of heads of arrows
vXQ1 = vXQ0 + sind(relative_heading(1:subsampling:end));
% Y coordinates of heads of arrows
vYQ1 = vYQ0 + cosd(relative_heading(1:subsampling:end));

% vector difference between heads & tails
vPx = (vXQ1 - vXQ0) * rMag;
vPy = (vYQ1 - vYQ0) * rMag;

% make plot 
h1 = plot (vX, vY, '.-','LineWidth',2); hold on;
% add arrows 
h2 = quiver (vXQ0,vYQ0, vPx, vPy, 0, 'r','LineWidth',1); grid on; hold off
axis equal
