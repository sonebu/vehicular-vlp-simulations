function [fim] = build_fim(delG_delP, sigma_W)
	num_observations = size(delG_delP, 1);
	num_estimations  = size(delG_delP, 2);
	fim = zeros(num_estimations, num_estimations);

	for mo=1:num_estimations
		for mp=1:num_estimations
			sum_elt = 0;
			for h=1:num_observations
				sum_elt = sum_elt + (1/(sigma_W(h).^2))*(delG_delP(h,mo)*delG_delP(h,mp));
			end
			fim(mo,mp) = -sum_elt;	
		end
	end
end