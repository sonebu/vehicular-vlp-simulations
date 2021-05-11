function [ dum ] = vehCfg_video( vehCfg )
%VEHCFG_VIDEO Summary of this function goes here
%   Detailed explanation goes here
dum = 0;
%%%%%%% VIDEO %%%%%%%
v=VideoWriter('test.avi');
v.FrameRate = 30;
open(v);
figure, hold on
for i=round(1:((1/vehCfg.t.dt)/v.FrameRate):length(vehCfg.target.tx1_qrx4.x))
    aa = plot(vehCfg.target.tx1_qrx4.x(1:i),vehCfg.target.tx1_qrx4.y(1:i),'g','LineWidth',1.5);
    aa = plot(vehCfg.target.tx2_qrx3.x(1:i),vehCfg.target.tx2_qrx3.y(1:i),'g','LineWidth',1.5);
    aa = plot(vehCfg.ego.tx1_qrx4.x(1:i),vehCfg.ego.tx1_qrx4.y(1:i),'r','LineWidth',1.5);
    aa = plot(vehCfg.ego.tx2_qrx3.x(1:i),vehCfg.ego.tx2_qrx3.y(1:i),'r','LineWidth',1.5);
    grid on
    xl = [min(min(vehCfg.target.tx1_qrx4.x),min(vehCfg.ego.tx1_qrx4.x))-1 max(max(vehCfg.target.tx1_qrx4.x),max(vehCfg.ego.tx1_qrx4.x))+1];
    yl = [min(min(vehCfg.target.tx1_qrx4.y),min(vehCfg.ego.tx1_qrx4.y))-1 max(max(vehCfg.target.tx1_qrx4.y),max(vehCfg.ego.tx1_qrx4.y))+1];
    szx = xl(2)-xl(1);
	szy = yl(2)-yl(1);
    if(szx)>=(szy)
        xlim(xl);
        ylim([yl(1)-(szx-szy)/2 yl(2)+(szx-szy)/2]);
    else
        xlim([xl(1)-(szy-szx)/2 xl(2)+(szy-szx)/2]);
        ylim(yl);
    end
%     drawnow
    frame = getframe(gcf);
    writeVideo(v,frame);
end
% close(frame)
close(v);
%%%%%%% VIDEO %%%%%%%


end

