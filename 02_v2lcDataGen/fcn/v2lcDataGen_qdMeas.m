function [A_sh, B_sh, C_sh, D_sh] = v2lcDataGen_qdMeas(z, y, x, dsL, dres, dpx,dr )
%v2lcDataGen_qdMeas Summary of this function goes here
%   Detailed explanation goes here

%%% OLD METHOD (Go over the whole quad, check if in circle). deleted for simplicity

%%% NEW METHOD (Go over the circle, check if quad)
% Find circle center on quad coordinate frame
teta1 = atan2(z,y);
teta2 = atan2(x,y);
qpd_y = tan(teta1)*dsL;
qpd_x = tan(teta2)*dsL;

if( (qpd_y==0)&&(dr<dpx) )
    tmp = 0;
    if(qpd_x>dr)
        tmp = 1;
    elseif(qpd_x<-dr)
        tmp = -1;
    else
        if(qpd_x+dr > dpx)
            tt = acos(qpd_x/dr);
            bb = qpd_x*dr*sin(tt);
            aa = acos(qpd_x/dr)*dr*dr - bb;
            alph = acos((dpx-(qpd_x))/dr);
            dd = alph*dr*dr - (dpx-(qpd_x))*dr*sin(alph);
            bbccdd = pi*dr*dr - aa;
            cc = bbccdd - bb - dd;
            tmp = (cc+bb-aa)/(bb+cc+aa);
        elseif(qpd_x-dr < -dpx)
            tt = acos(-qpd_x/dr);
            bb = -qpd_x*dr*sin(tt);
            aa = acos(-qpd_x/dr)*dr*dr - bb;
            alph = acos((dpx-(-qpd_x))/dr);
            dd = alph*dr*dr - (dpx-(-qpd_x))*dr*sin(alph);
            bbccdd = pi*dr*dr - aa;
            cc = bbccdd - bb - dd;
            tmp = -(cc+bb-aa)/(bb+cc+aa);
        else
            tmp = (pi*dr*dr - 2*(acos(qpd_x/dr)*dr*dr - qpd_x*dr*sin(acos(qpd_x/dr))))/(pi*dr*dr);
        end
    end
    A_sh = (2-1-tmp)/4;
    B_sh = (1+tmp)/4;
    C_sh = (2-1-tmp)/4;
    D_sh = (1+tmp)/4;
else
    % Generate bounds for looping
    srch_x_neg = qpd_x-dr;
    srch_x_pos = qpd_x+dr;
    x_ar = srch_x_neg:dres:srch_x_pos;
    y_ar_pos = sqrt(abs(dr.^2-(x_ar-qpd_x).^2))+qpd_y;
    y_ar_neg = -sqrt(abs(dr.^2-(x_ar-qpd_x).^2))+qpd_y;
    
    x_filt = (x_ar <= dpx) & (x_ar >= -dpx);
    x_vals=x_ar(x_filt);
    
    y_p_vals = y_ar_pos(x_filt);
    y_p_vals(y_p_vals>dpx) = dpx;
    y_p_vals(y_p_vals<-dpx) = -dpx;
    y_n_vals = y_ar_neg(x_filt);
    y_n_vals(y_n_vals>dpx) = dpx;
    y_n_vals(y_n_vals<-dpx) = -dpx;
    
    A_ctr = 0; B_ctr = 0; C_ctr = 0; D_ctr = 0;
    for i=1:length(x_vals)
        
        y_vals = y_n_vals(i):dres:y_p_vals(i);
        % Works for uniform only! But for the spot size we use, uniform is
        % a good assumption, see Manojlovic's QPD sensitivity paper for
        % more information.
        if(x_vals(i) > 0)
            s1 = y_vals > 0;
            a = sum(s1);
            B_ctr = B_ctr + a;
            D_ctr = D_ctr + length(s1) - a;
        else
            s1 = y_vals > 0;
            a = sum(s1);
            A_ctr = A_ctr + a;
            C_ctr = C_ctr + length(s1) - a;
        end
        
    end
    total_ctr = pi*(dr^2)/(dres^2);
    A_sh = A_ctr/total_ctr;
    B_sh = B_ctr/total_ctr;
    C_sh = C_ctr/total_ctr;
    D_sh = D_ctr/total_ctr;
    
end

end

