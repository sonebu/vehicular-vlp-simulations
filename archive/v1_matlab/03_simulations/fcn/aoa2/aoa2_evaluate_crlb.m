function [crlb] = aoa2_evaluate_crlb(sigma_W, P, L)
	num_estimations  = 4;
	num_observations = 4;
	delG_delP = zeros(num_observations, num_estimations);

	x11 = P(1); y11 = P(2); x12 = P(3); y12 = P(4);

	delG_delP(1,1) =  y11/(x11.^2 + y11.^2);
	delG_delP(1,2) = -x11/(x11.^2 + y11.^2);
	delG_delP(1,3) =  0;
	delG_delP(1,4) =  0;
	delG_delP(2,1) =  0;
	delG_delP(2,2) =  0;
	delG_delP(2,3) =  y12/(x12.^2 + y12.^2);
	delG_delP(2,4) = -x12/(x12.^2 + y12.^2);
	delG_delP(3,1) =  y11/((x11-L).^2 + y11.^2);
	delG_delP(3,2) = -(x11-L)/((x11-L).^2 + y11.^2);
	delG_delP(3,3) =  0;
	delG_delP(3,4) =  0;
	delG_delP(4,1) =  0;
	delG_delP(4,2) =  0;
	delG_delP(4,3) =  y12/((x12-L).^2 + y12.^2);
	delG_delP(4,4) = -(x12-L)/((x12-L).^2 + y12.^2);

	fim = build_fim(delG_delP, sigma_W);
	crlb = inv(fim);
end