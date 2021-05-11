clear all;
close all;
clc;
addpath('fcn/aoa2');
addpath('fcn/old');
addpath('fcn/common');

data_file = '../02_v2lcDataGen/data/v2lcRun_sm4_3lane_p1.mat';
T_s = 1/1e6; % sec
circuit_temperature = 298;
iterations = 100;
validAoAThd = 1e-7;
add_noise = 1;

sim = 0;
res = 1;
c1_arr = {'day'};
c2_arr = {'50Hz'};
c3_arr = {'rain'};

if sim == 1
    for i_c1=1:length(c1_arr)
        for i_c2=1:length(c2_arr)
            for i_c3=1:length(c3_arr)
                cond1 = c1_arr{i_c1};
                cond2 = c2_arr{i_c2};
                cond3 = c3_arr{i_c3};
                res_file  = strcat('results/aoa2_sm4_',cond1,'_',cond2,'_',cond3,'_p1.mat');
                
                aoa2_simulate(data_file, res_file, iterations, validAoAThd, add_noise, T_s, circuit_temperature, cond1, cond2, cond3);
            end
        end
    end
end

%% results
if res == 1
    cond1 = 'day';
    cond2 = '50Hz';
    cond3 = 'rain';
    
    res_file  = strcat('results/aoa2_sm4_',cond1,'_',cond2,'_',cond3,'_total.mat');
    load(res_file)
    
    ex11 = zeros(length(vx11_total),size(cx11_total,2));
    ey11 = zeros(length(vx11_total),size(cx11_total,2));
    ex12 = zeros(length(vx11_total),size(cx11_total,2));
    ey12 = zeros(length(vx11_total),size(cx11_total,2));
    for ik=1:length(vx11_total)
        k = floor(ik/vlpDecimRate); % kth_estimation
        if(k==0)
            ex11(ik) = NaN;
            ey11(ik) = NaN;
            ex12(ik) = NaN;
            ey12(ik) = NaN;
        else
            ex11(ik,:) = ones(size(cx11_total(k,:)))*vx11_total(ik) - cx11_total(k,:);
            ey11(ik,:) = ones(size(cy11_total(k,:)))*vy11_total(ik) - cy11_total(k,:);
            ex12(ik,:) = ones(size(cx12_total(k,:)))*vx12_total(ik) - cx12_total(k,:);
            ey12(ik,:) = ones(size(cy12_total(k,:)))*vy12_total(ik) - cy12_total(k,:);
        end
    end
    ex11_stdev = std(ex11,0,2); ey11_stdev = std(ey11,0,2); ex12_stdev = std(ex12,0,2); ey12_stdev = std(ey12,0,2);
    e4d_stdev = sqrt(ex11_stdev.^2+ey11_stdev.^2+ex12_stdev.^2+ey12_stdev.^2);
    
    err_z_tot = [];
    
    target_front_center_x_relative  = -3:0.25:3;
    target_front_center_y_relative  = [1:0.25:2.5] + vehicle.target.length;
    target_heading_absolute         = -5:2.5:5;
    
    err_z = zeros(length(target_front_center_x_relative),length(target_front_center_y_relative));
    for ix = 1:length(target_front_center_x_relative)
        for iy = 1:length(target_front_center_y_relative)
            err_avg = 0;
            for ih = 1:length(target_heading_absolute)
                ii = 1 + (ih - 1) + length(target_heading_absolute)*(iy-1) + length(target_front_center_y_relative)*length(target_heading_absolute)*(ix-1);
                new_err = e4d_stdev(ii);
                err_avg = (err_avg*(ih-1) + new_err)/ih;
            end
            err_z(ix,iy) = err_avg;
        end
    end
    
    ii_leftoff = ii;
    err_z_tot = [err_z_tot err_z];
    
    for i_tt = 2:6
        target_front_center_x_relative  = -3:0.25:3;
        target_front_center_y_relative  = [(2.75+(i_tt-2)*2.5):0.25:(5+(i_tt-2)*2.5)] + vehicle.target.length;
        target_heading_absolute         = -5:2.5:5;
        
        err_z = zeros(length(target_front_center_x_relative),length(target_front_center_y_relative));
        for ix = 1:length(target_front_center_x_relative)
            for iy = 1:length(target_front_center_y_relative)
                err_avg = 0;
                for ih = 1:length(target_heading_absolute)
                    ii = ii_leftoff + 1 + (ih - 1) + length(target_heading_absolute)*(iy-1) + length(target_front_center_y_relative)*length(target_heading_absolute)*(ix-1);
                    new_err = e4d_stdev(ii);
                    err_avg = (err_avg*(ih-1) + new_err)/ih;
                end
                err_z(ix,iy) = err_avg;
            end
        end
        
        ii_leftoff = ii;
        err_z_tot = [err_z_tot err_z];
    end
    err_z_tot_dayrain = err_z_tot;

    clearvars -except err_z_tot_dayrain
    
    cond1 = 'night';
    cond2 = '50Hz';
    cond3 = 'clear';
    
    res_file  = strcat('results/aoa2_sm4_',cond1,'_',cond2,'_',cond3,'_total.mat');
    load(res_file)
    
    ex11 = zeros(length(vx11_total),size(cx11_total,2));
    ey11 = zeros(length(vx11_total),size(cx11_total,2));
    ex12 = zeros(length(vx11_total),size(cx11_total,2));
    ey12 = zeros(length(vx11_total),size(cx11_total,2));
    for ik=1:length(vx11_total)
        k = floor(ik/vlpDecimRate); % kth_estimation
        if(k==0)
            ex11(ik) = NaN;
            ey11(ik) = NaN;
            ex12(ik) = NaN;
            ey12(ik) = NaN;
        else
            ex11(ik,:) = ones(size(cx11_total(k,:)))*vx11_total(ik) - cx11_total(k,:);
            ey11(ik,:) = ones(size(cy11_total(k,:)))*vy11_total(ik) - cy11_total(k,:);
            ex12(ik,:) = ones(size(cx12_total(k,:)))*vx12_total(ik) - cx12_total(k,:);
            ey12(ik,:) = ones(size(cy12_total(k,:)))*vy12_total(ik) - cy12_total(k,:);
        end
    end
    ex11_stdev = std(ex11,0,2); ey11_stdev = std(ey11,0,2); ex12_stdev = std(ex12,0,2); ey12_stdev = std(ey12,0,2);
    e4d_stdev = sqrt(ex11_stdev.^2+ey11_stdev.^2+ex12_stdev.^2+ey12_stdev.^2);
    
    err_z_tot = [];
    
    target_front_center_x_relative  = -3:0.25:3;
    target_front_center_y_relative  = [1:0.25:2.5] + vehicle.target.length;
    target_heading_absolute         = -5:2.5:5;
    
    err_z = zeros(length(target_front_center_x_relative),length(target_front_center_y_relative));
    for ix = 1:length(target_front_center_x_relative)
        for iy = 1:length(target_front_center_y_relative)
            err_avg = 0;
            for ih = 1:length(target_heading_absolute)
                ii = 1 + (ih - 1) + length(target_heading_absolute)*(iy-1) + length(target_front_center_y_relative)*length(target_heading_absolute)*(ix-1);
                new_err = e4d_stdev(ii);
                err_avg = (err_avg*(ih-1) + new_err)/ih;
            end
            err_z(ix,iy) = err_avg;
        end
    end
    
    ii_leftoff = ii;
    err_z_tot = [err_z_tot err_z];
    
    for i_tt = 2:6
        target_front_center_x_relative  = -3:0.25:3;
        target_front_center_y_relative  = [(2.75+(i_tt-2)*2.5):0.25:(5+(i_tt-2)*2.5)] + vehicle.target.length;
        target_heading_absolute         = -5:2.5:5;
        
        err_z = zeros(length(target_front_center_x_relative),length(target_front_center_y_relative));
        for ix = 1:length(target_front_center_x_relative)
            for iy = 1:length(target_front_center_y_relative)
                err_avg = 0;
                for ih = 1:length(target_heading_absolute)
                    ii = ii_leftoff + 1 + (ih - 1) + length(target_heading_absolute)*(iy-1) + length(target_front_center_y_relative)*length(target_heading_absolute)*(ix-1);
                    new_err = e4d_stdev(ii);
                    err_avg = (err_avg*(ih-1) + new_err)/ih;
                end
                err_z(ix,iy) = err_avg;
            end
        end
        
        ii_leftoff = ii;
        err_z_tot = [err_z_tot err_z];
    end
    err_z_tot_nightclear = err_z_tot;

    target_front_center_x_relative  = -3:0.25:3;
    target_front_center_y_relative  = [1:0.25:15] + vehicle.target.length;
    target_heading_absolute         = -5:2.5:5;
    
    err_z_tot_nightclear(find(err_z_tot_nightclear>10)) = 10;
    err_z_tot_dayrain(find(err_z_tot_dayrain>10)) = 10;
    
    xv = target_front_center_x_relative;
    yv = target_front_center_y_relative;
    [xx, yy] = meshgrid(xv,yv);
    figure
    pl = pcolor(xx',yy'-vehicle.target.length,log10(err_z_tot_dayrain));
    pl.FaceColor ='interp';
    pl.EdgeColor ='none';
    cb = colorbar;
    log_list = [0.01 0.1 1 10];
    cb.Ticks=log10(log_list);
    cb.TickLabels={'0.01','0.1','1','10'};
    cb.Title.String='Error (m)';
    cb.Limits=[-2.3 1.3];
    axis equal
    axis tight
    ylim([-0.5 15])
    xlim([-3 3])
    figure
    pl = pcolor(xx',yy'-vehicle.target.length,log10(err_z_tot_nightclear));
    pl.FaceColor ='interp';
    pl.EdgeColor ='none';
    cb = colorbar;
    log_list = [0.01 0.1 1 10];
    cb.Ticks=log10(log_list);
    cb.TickLabels={'0.01','0.1','1','10'};
    cb.Title.String='Error (m)';
    cb.Limits=[-2.3 1.3];
    axis equal
    axis tight
    ylim([-0.5 15])
    xlim([-3 3])
end

