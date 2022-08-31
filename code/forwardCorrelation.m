function varargout = forwardCorrelation(Stim, Robs, win, inds, nbasis, validonly, normalize)
% varargout = forwardCorrelation(Stim, Robs, win, inds)

% % get good indices
% ecc = hypot(eyeAtFrame(:,2)-Exp.S.centerPix(1), eyeAtFrame(:,3)-Exp.S.centerPix(2))/Exp.S.pixPerDeg;
% ix = ecc < 5.2 & labels == 1;
% ix = sum(conv2(double(ix), eye(nlags), 'full'),2) == nlags; % all timelags good
% 

NT = size(Stim, 1);
if nargin < 7 || isempty(normalize)
    normalize = false;
end

if nargin < 6 || isempty(validonly)
    validonly = true;
end

if nargin < 4 || isempty(inds)
    inds = (1:NT)'; 
end

% convert to logical indices
ix = false(NT,1);
ix(inds) = true;

% compute STA with forward correlation
NC = size(Robs,2);
nlags = diff(win) + 1;

if nargin < 5 || isempty(nbasis)
   nbasis = nlags; 
end


Rdelta = Robs - mean(Robs);

if nargout > 1
    getebars = true;
else
    getebars = false;
end

if nbasis==nlags
    B = eye(nlags);
else
    % tent basis
    bs = ceil(nlags/nbasis);
    ctrs = win(1):bs:win(2);
    lags = win(1):win(2);
    xdiff = abs(lags(:)-ctrs(:)');
    B = max(1-xdiff/bs,0);
end

% only use indices where all lags are valid
if validonly
    ix = abs(sum(conv2(double(ix), B, 'full'), 2) - sum(B(:)))<.1; % all timelags good
end
ix = find(ix);

dims = size(Stim,2);

stas = zeros(nlags, dims, NC);
if getebars
    stasSd = zeros(nlags, dims, NC);
end

disp('Running forward correlation...')
n = zeros(dims, nlags);
for ii = 1:dims
%     fprintf('%d/%d\n', ii, dims)
        Xstim = conv2(Stim(:,ii), B, 'full');
        Xstim = Xstim(1:end-nlags+1,:);
        if win(1)~=0
            Xstim = circshift(Xstim, win(1)); % shift back by the pre-stimulus lags
        end
        
        n(ii,:) = sum(Xstim(ix,:))*B';
        
        if getebars
            for cc = 1:NC
                x = (Xstim(ix,:).*Rdelta(ix,cc))*B';
                stas(:, ii, cc) = sum(x);
                stasSd(:,ii,cc) = std(x);
            end
        else
            stas(:, ii, :) = B*(Xstim(ix,:)'*Rdelta(ix,:));
        end
        
end
disp('Done')

if normalize
    % normalize and get rid of NANS
    for cc = 1:NC
        x = stas(:,:,cc);% ./ n';
        x(n'==0) = 0;
        x = x .* (n'./max(n(:)));
        stas(:,:,cc) =  x;
    end
end

varargout{1} = stas;
if nargout > 1
    varargout{2} = stasSd;
end

if nargout > 2
    varargout{3} = n;
end