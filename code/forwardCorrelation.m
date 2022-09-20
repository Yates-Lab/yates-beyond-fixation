function varargout = forwardCorrelation(Stim, Robs, win, inds, nbasis, validonly, normalize)
% varargout = forwardCorrelation(Stim, Robs, win, inds)

% % get good indices
% ecc = hypot(eyeAtFrame(:,2)-Exp.S.centerPix(1), eyeAtFrame(:,3)-Exp.S.centerPix(2))/Exp.S.pixPerDeg;
% ix = ecc < 5.2 & labels == 1;
% ix = sum(conv2(double(ix), eye(nlags), 'full'),2) == nlags; % all timelags good
% 
if issparse(Stim) && issparse(Robs)
    usesparse = true;
    disp('use sparse')
else
    usesparse = false;
end

NT = size(Stim, 1);
if nargin < 7 || isempty(normalize)
    normalize = true;
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

if usesparse
    Rdelta = Robs;
else
    Rdelta = Robs - mean(Robs);
end

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

if usesparse
    B = sparse(B);
end

% only use indices where all lags are valid
if validonly
    if usesparse
        ix = abs(sum(sconv2(sparse(double(ix)), B, 'full'), 2) - sum(B(:)))<.1; % all timelags good
    else
        ix = abs(sum(conv2(double(ix), B, 'full'), 2) - sum(B(:)))<.1; % all timelags good
    end
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
        if usesparse
            Xstim = sconv2(Stim(:,ii), B, 'full');
        else
            Xstim = conv2(Stim(:,ii), B, 'full');
        end

        Xstim = Xstim(1:end-nlags+1,:);
        if win(1)~=0
            Xstim = circshift(Xstim, win(1)); % shift back by the pre-stimulus lags
        end
        
        n(ii,:) = sum(Xstim(ix,:))*B';
        
        if getebars
%             for cc = 1:NC
%                 x = (Xstim(ix,:).*Rdelta(ix,cc))*B';
%                 stas(:, ii, cc) = sum(x);
%                 stasSd(:,ii,cc) = std(x);
%             end
            stas(:, ii, :) = B*(Xstim(ix,:)'*Rdelta(ix,:));
            stasSd(:,ii,:) = B*(Xstim(ix,:).^2'*Rdelta(ix,:).^2);
        else
            stas(:, ii, :) = B*(Xstim(ix,:)'*Rdelta(ix,:));
        end
        
end
disp('Done')

if normalize
    % normalize and get rid of NANS
    n = max(n, 1);

    for cc = 1:NC
        stas(:,:,cc) = stas(:,:,cc) ./ n';
        if getebars
            stasSd(:,:,cc) = stasSd(:,:,cc)./ n';
            stasSd(:,:,cc) = stasSd(:,:,cc) - stas(:,:,cc).^2;
        end
%         x = stas(:,:,cc);% ./ n';
%         x(n'==0) = 0;
%         x = x .* (n'./max(n(:)));
%         stas(:,:,cc) =  x;
         
    end
end

varargout{1} = stas;
if nargout > 1
    varargout{2} = stasSd;
end

if nargout > 2
    varargout{3} = n;
end