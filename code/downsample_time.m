function y = downsample_time( x, ds )
% Usage: y = downsample_time( x, ds )
% only works with max of 2-dimensions for x. Assumes time is first axis

[NTold,dims] = size(x);
% flipped = 0;
% if dims > NTold
% 	% then assume flipped
% 	flipped = 1;
% 	x = x';
% end

NTnew = floor(NTold/ds);

y = zeros(NTnew, dims);
for nn = 1:ds
	y = y + x(nn+(0:NTnew-1)*ds,:);
end
