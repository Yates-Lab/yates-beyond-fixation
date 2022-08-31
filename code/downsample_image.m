function imn = downsample_image( imo, ds, centered )
%
% Usage: im_new = downsample_image( im_orig, ds, <centered> )

if nargin < 3 
	centered = 0;
end

if length(size(imo)) == 3
	if size(imo,3) > 1
		error('Can only handle one image at a time.')
	else
		imo = squeeze(imo);
	end
end

[NY, NX] = size(imo);

NNY = floor(NY/ds);
NNX = floor(NX/ds);

if centered
	xoff = floor((NX-NNX*ds)/2);
	yoff = floor((NY-NNY*ds)/2);
else
	xoff = 0; yoff = 0;
end	

imn = zeros( NNY, NNX );
for ii = 1:ds
	for jj = 1:ds
		imn = imn + imo( (0:NNY-1)*ds+jj + yoff, (0:NNX-1)*ds+ii + xoff );
	end
end

imn = imn / ds^2;
