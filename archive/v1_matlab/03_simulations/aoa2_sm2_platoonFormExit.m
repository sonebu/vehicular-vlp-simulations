clear all;
close all; 
clc;
addpath('fcn/aoa2');
addpath('fcn/old');
addpath('fcn/common');

data_file = '../02_v2lcDataGen/data/v2lcRun_sm2_platoonFormExit.mat';
T_s = 1/1e6; % sec
circuit_temperature = 298;
iterations = 100;
validAoAThd = 1e-7;
add_noise = 1;

sim = 0;
res = 1;
c1_arr = {'day','night'};
c2_arr = {'100Hz'};
c3_arr = {'clear','rain','fog'};

if sim == 1
    for i_c1=1:length(c1_arr)
        for i_c2=1:length(c2_arr)
            for i_c3=1:length(c3_arr)
                cond1 = c1_arr{i_c1};
                cond2 = c2_arr{i_c2};
                cond3 = c3_arr{i_c3};
                res_file  = strcat('results/aoa2_sm2_',cond1,'_',cond2,'_',cond3,'.mat');
                
                aoa2_simulate(data_file, res_file, iterations, validAoAThd, add_noise, T_s, circuit_temperature, cond1, cond2, cond3);
            end
        end
    end
end

%% results
if res == 1
    h = plot(1:10,1:10,1:10,1:10,1:10,1:10,1:10,1:10,1:10,1:10,1:10,2:11);
    colors = get(h,'Color');
    color_counter = 1;
    close; 
    clear h
    figure
    for ic1=1:length(c1_arr)
        for ic2=1:length(c2_arr)
            for ic3=1:length(c3_arr)
                cond1 = c1_arr{ic1};
                cond2 = c2_arr{ic2};
                cond3 = c3_arr{ic3};
                res_file  = strcat('results/aoa2_sm2_',cond1,'_',cond2,'_',cond3,'.mat');
                
                load(res_file)
                
%                 figure
%                 plot_dir(vehicle.target_relative.tx1_qrx4.x,vehicle.target_relative.tx1_qrx4.y,vehicle.target_relative.heading, 15)
%                 hold on
%                 post = 4;
%                 limits_xp = vehicle.target_relative.tx1_qrx4.x(end-post)-10*sind(20+vehicle.target_relative.heading(end-post));
%                 limits_xn = vehicle.target_relative.tx1_qrx4.x(end-post)+10*sind(20-vehicle.target_relative.heading(end-post));
%                 limits_yp = vehicle.target_relative.tx1_qrx4.y(end-post)-10*cosd(20+vehicle.target_relative.heading(end-post));
%                 limits_yn = vehicle.target_relative.tx1_qrx4.y(end-post)-10*cosd(20-vehicle.target_relative.heading(end-post));
%                 plot([vehicle.target_relative.tx1_qrx4.x(end-post) limits_xp],[vehicle.target_relative.tx1_qrx4.y(end-post) limits_yp],'g--','LineWidth',2)
%                 plot([vehicle.target_relative.tx1_qrx4.x(end-post) limits_xn],[vehicle.target_relative.tx1_qrx4.y(end-post) limits_yn],'g--','LineWidth',2)
%                 axis equal; ylim([ -1 10 ]); xlim([ -4 8 ])
                
                ex11 = zeros(length(vehicle.target_relative.tx1_qrx4.x),size(cx11,2));
                ey11 = zeros(length(vehicle.target_relative.tx1_qrx4.x),size(cx11,2));
                ex12 = zeros(length(vehicle.target_relative.tx1_qrx4.x),size(cx11,2));
                ey12 = zeros(length(vehicle.target_relative.tx1_qrx4.x),size(cx11,2));
                ex11_maxpool = zeros(size(cx11));     ex11_avgpool = zeros(size(cx11));
                ey11_maxpool = zeros(size(cx11));     ey11_avgpool = zeros(size(cx11));
                ex12_maxpool = zeros(size(cx11));     ex12_avgpool = zeros(size(cx11));
                ey12_maxpool = zeros(size(cx11));     ey12_avgpool = zeros(size(cx11));
                k_prev = 0;
                for ik=1:length(vehicle.target_relative.tx1_qrx4.x)
                    k = floor(ik/vlpDecimRate); % kth_estimation
                    if(k==0)
                        ex11(ik) = NaN;
                        ey11(ik) = NaN;
                        ex12(ik) = NaN;
                        ey12(ik) = NaN;
                    else
                        ex11(ik,:) = ones(size(cx11(k,:)))*vehicle.target_relative.tx1_qrx4.x(ik) - cx11(k,:);
                        ey11(ik,:) = ones(size(cy11(k,:)))*vehicle.target_relative.tx1_qrx4.y(ik) - cy11(k,:);
                        ex12(ik,:) = ones(size(cx12(k,:)))*vehicle.target_relative.tx2_qrx3.x(ik) - cx12(k,:);
                        ey12(ik,:) = ones(size(cy12(k,:)))*vehicle.target_relative.tx2_qrx3.y(ik) - cy12(k,:);
                        if(k_prev~=k) && (k_prev~=0)
                            ex11_maxpool(k,:) = max(ex11((1 + ik - vlpDecimRate):ik,:),[],1);
                            ey11_maxpool(k,:) = max(ey11((1 + ik - vlpDecimRate):ik,:),[],1);
                            ex12_maxpool(k,:) = max(ex12((1 + ik - vlpDecimRate):ik,:),[],1);
                            ey12_maxpool(k,:) = max(ey12((1 + ik - vlpDecimRate):ik,:),[],1);
                            ex11_avgpool(k,:) = mean(ex11((1 + ik - vlpDecimRate):ik,:),1);
                            ey11_avgpool(k,:) = mean(ey11((1 + ik - vlpDecimRate):ik,:),1);
                            ex12_avgpool(k,:) = mean(ex12((1 + ik - vlpDecimRate):ik,:),1);
                            ey12_avgpool(k,:) = mean(ey12((1 + ik - vlpDecimRate):ik,:),1);
                        end
                        k_prev = k;
                    end
                end
                ex11_mean = mean(ex11,2);   ey11_mean = mean(ey11,2);   ex12_mean = mean(ex12,2);   ey12_mean = mean(ey12,2);
                ex11_stdev = std(ex11,0,2); ey11_stdev = std(ey11,0,2); ex12_stdev = std(ex12,0,2); ey12_stdev = std(ey12,0,2);
                e4d_mean = sqrt(ex11_mean.^2+ey11_mean.^2+ex12_mean.^2+ey12_mean.^2);
                e4d_stdev = sqrt(ex11_stdev.^2+ey11_stdev.^2+ex12_stdev.^2+ey12_stdev.^2);
                e2d_mean = sqrt(ex11_mean.^2+ey11_mean.^2);
                e2d_stdev = sqrt(ex11_stdev.^2+ey11_stdev.^2);
                ex11_maxpool_mean = mean(ex11_maxpool,2);    ex11_maxpool_stdev = std(ex11_maxpool,0,2);
                ey11_maxpool_mean = mean(ey11_maxpool,2);    ey11_maxpool_stdev = std(ey11_maxpool,0,2);
                ex12_maxpool_mean = mean(ex12_maxpool,2);    ex12_maxpool_stdev = std(ex12_maxpool,0,2);
                ey12_maxpool_mean = mean(ey12_maxpool,2);    ey12_maxpool_stdev = std(ey12_maxpool,0,2);
                e4d_maxpool_mean = sqrt(ex11_maxpool_mean.^2+ey11_maxpool_mean.^2+ex12_maxpool_mean.^2+ey12_maxpool_mean.^2);
                e4d_maxpool_stdev = sqrt(ex11_maxpool_stdev.^2+ey11_maxpool_stdev.^2+ex12_maxpool_stdev.^2+ey12_maxpool_stdev.^2);
                ex11_avgpool_mean = mean(ex11_avgpool,2);    ex11_avgpool_stdev = std(ex11_avgpool,0,2);
                ey11_avgpool_mean = mean(ey11_avgpool,2);    ey11_avgpool_stdev = std(ey11_avgpool,0,2);
                ex12_avgpool_mean = mean(ex12_avgpool,2);    ex12_avgpool_stdev = std(ex12_avgpool,0,2);
                ey12_avgpool_mean = mean(ey12_avgpool,2);    ey12_avgpool_stdev = std(ey12_avgpool,0,2);
                e4d_avgpool_mean = sqrt(ex11_avgpool_mean.^2+ey11_avgpool_mean.^2+ex12_avgpool_mean.^2+ey12_avgpool_mean.^2);
                e4d_avgpool_stdev = sqrt(ex11_avgpool_stdev.^2+ey11_avgpool_stdev.^2+ex12_avgpool_stdev.^2+ey12_avgpool_stdev.^2);
                
                aoa2_crlb = zeros(size(cx11,1),4,4);
                for i_crlb=1:size(cx11,1)
                    if(i_crlb == 1)
                        aoa2_crlb(i_crlb,:,:) = NaN;
                    else
                        sampler = vlpDecimRate;
                        x11 = mean(vehicle.target_relative.tx1_qrx4.x( ((i_crlb-1)*sampler+1):(i_crlb*sampler) ));
                        y11 = mean(vehicle.target_relative.tx1_qrx4.y( ((i_crlb-1)*sampler+1):(i_crlb*sampler) ));
                        x12 = mean(vehicle.target_relative.tx2_qrx3.x( ((i_crlb-1)*sampler+1):(i_crlb*sampler) ));
                        y12 = mean(vehicle.target_relative.tx2_qrx3.y( ((i_crlb-1)*sampler+1):(i_crlb*sampler) ));
                        
                        sigmaW       = deg2rad([std(aoa_11,0,2) std(aoa_12,0,2) std(aoa_21,0,2) std(aoa_22,0,2)]);
                        P            = [x11 y11 x12 y12];
                        L            = vehicle.ego.width;
                        aoa2_crlb(i_crlb,:,:) = aoa2_evaluate_crlb(sigmaW(i_crlb,:), P, L);
                    end
                end
                crlb_4d = sqrt(abs(aoa2_crlb(:,1,1)) + abs(aoa2_crlb(:,2,2)) + abs(aoa2_crlb(:,3,3)) + abs(aoa2_crlb(:,4,4)) );
                crlb_2d = sqrt(abs(aoa2_crlb(:,1,1)) + abs(aoa2_crlb(:,2,2)) );

                semilogy(vehicle.t.values(1:vlpDecimRate:end),e4d_maxpool_mean,'LineWidth',2,'Color',colors{color_counter})
                hold on
                ylim([1e-2 1e0]), grid on
                color_counter = color_counter + 1;
            end
        end
    end
    
    legend('day, clear','day, rain','day, fog','night, clear','night, rain','night, fog','Location','best')
end

