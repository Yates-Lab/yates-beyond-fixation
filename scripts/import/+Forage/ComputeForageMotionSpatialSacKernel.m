function rfinfo = ComputeForageMotionSpatialSacKernel2(StimX,StimPX,SacX,StimY,GRID,FileTag)
% function rfinfo = ComputeForageSpatialKernel(Exp)
%   input: takes StimX and StimY (stim history and spike counts)
%     StimX: [Nframes,3 + NT] - fields, matlab time, ephys time, stim, (if stim)
%     SacX:  [Nframes,3] - amplitude, int direction, angle of sac onset
%     StimY: [Nframes,1] - spike counts per video frame
%
%  rfinfo = returns all computed information, for use to make figures
%
%******** use reverse correlation of stim vs no stim to find temporal

rfinfo = [];

%******** the temporal kernel from the data stream
SampRate = median(diff(StimX(:,1)));  % median screen flip time
SN = size(StimX,1);
SDTA = -20;   % stim frames, relative to stim onset, must be negative
SDTB = 30;   % stim frames after onset, must be positive 
SXX = SDTA:SDTB;
tSXX = SXX * (SampRate * 1000);  % put into ms
POSTPERIOD = [41,120];
DURPERIOD = [0,40];
PREPERIOD = [-150,0];

%**** transform some parameters of the grid
DTA = -5;   % stim frames, relative to stim onset, must be negative
DTB = 18;   % stim frames after onset, must be positive 
XX = DTA:DTB;
tXX = XX * (SampRate * 1000);  % put into ms
KN = 10;
%*******
TT = size(XX,2);
Nx = floor(GRID.box(3)/GRID.div);
Ny = floor(GRID.box(4)/GRID.div);
NT = (Nx * Ny);
BONFP = (0.05/NT);
%******
BaseX = GRID.box(1) - (GRID.box(3)/2);
BaseY = GRID.box(2) - (GRID.box(4)/2);
Div = GRID.div;
Zx = BaseX -(Div/2) + (1:Nx)*Div;
Zy = BaseY - (Div/2) + (1:Ny)*Div;
ND = 16; %directions of motion
SACN = 16;  % directions of saccade
%*** check for varying sparesness in stimulus
spark = unique(StimX(:,3))';
NS = length(spark);
%********
spud = unique(StimPX(:,4:end));  % all integers but not zero
NP = length(spud)-1;
%********

%***** plot saccade modulation functions
mvec = zeros(SACN+1,(SDTB-SDTA+1));
mvec2 = zeros(SACN+1,(SDTB-SDTA+1));
mcnt = zeros(SACN+1,1);
z = find(SacX(:,1) > 0);
SC = size(z,1);
for k = 1:SC
   kk = z(k);
   ampo = SacX(kk,1);
   diro = SacX(kk,2);
   if (ampo < 20) && (ampo > 0)
     if ((kk+SDTA) > 0) && ((kk+SDTB) < SN)
        fr = StimY((kk+SDTA):(kk+SDTB)) * (1/SampRate);
        mvec(diro,:) = mvec(diro,:) + fr';
        mvec(SACN+1,:) = mvec(SACN+1,:) + fr';
        mvec2(diro,:) = mvec2(diro,:) + (fr.^2)';
        mvec2(SACN+1,:) = mvec2(SACN+1,:) + (fr.^2)';
        mcnt(diro,1) = mcnt(diro,1) + 1;
        mcnt(SACN+1,1) = mcnt(SACN+1,1) + 1;
     end
   end
end
for k = 1:size(mvec,1)
    if (mcnt(k,1) > 0)
        mvec(k,:) = mvec(k,:)/mcnt(k,1);
        mvec2(k,:) = sqrt( ((mvec2(k,:)/mcnt(k,1)) - (mvec(k,:).^2))/mcnt(k,1));      
    end
end
sacbase = mean(mvec(SACN+1,1:-SDTA));
sacbasestd = mean(mvec2(SACN+1,1:-SDTA));
postx = find( (tSXX >= POSTPERIOD(1)) & (tSXX < POSTPERIOD(2)) );
postcurve = mean( mvec(1:16,postx)' );
postcurve_std = mean( mvec2(1:16,postx)' ) / sqrt(length(postx));
durx = find( (tSXX >= DURPERIOD(1)) & (tSXX < DURPERIOD(2)) );
durcurve = mean( mvec(1:16,durx)' );
durcurve_std = mean( mvec2(1:16,durx)' ) / sqrt(length(durx));
prex = find( (tSXX >= PREPERIOD(1)) & (tSXX < PREPERIOD(2)) );
precurve = mean( mvec(1:16,prex)' );
precurve_std = mean( mvec2(1:16,prex)' ) / sqrt(length(prex));

%****** preserve necessary rfinfo for plotting later
rfinfo.SampRate = SampRate;
rfinfo.tSXX = tSXX;
rfinfo.SACN = SACN;
rfinfo.mvec = mvec;
rfinfo.mvec2 = mvec2;
rfinfo.POSTPERIOD = POSTPERIOD;
rfinfo.DURPERIOD = DURPERIOD;
rfinfo.PREPERIOD = PREPERIOD;
rfinfo.sacbase = sacbase;
rfinfo.sacbasestd = sacbasestd;
rfinfo.ND = ND;
rfinfo.postcurve = postcurve;
rfinfo.postcurve_std = postcurve_std;
rfinfo.durcurve = durcurve;
rfinfo.durcurve_std = durcurve_std;
rfinfo.precurve = precurve;
rfinfo.precurve_std = precurve_std;
%**********

%******* compute RFs and significant locations
zmcounts = cell(1,ND);
zmcounts2 = cell(1,ND);
zcounts = cell(1,ND);
zmsem = cell(1,ND);
for k = 1:ND
    zmcounts{k} = zeros(NT,TT);
    zmcounts2{k} = zeros(NT,TT);
    zcounts{k} = zeros(NT,1);
    zmsem{k} = zeros(NT,TT);
end
%******* compute RFs and significant locations
zpcounts = cell(1,NP);
zpcounts2 = cell(1,NP);
ppcounts = cell(1,NP);
zpsem = cell(1,NP);
for k = 1:NP
    zpcounts{k} = zeros(NT,TT);
    zpcounts2{k} = zeros(NT,TT);
    ppcounts{k} = zeros(NT,1);
    zpsem{k} = zeros(NT,TT);
end
%******* compute RFs and significant locations
if (NS > 1)
  pmcounts = cell(1,NS);
  pmcounts2 = cell(1,NS);
  pcounts = cell(1,NS);
  pmsem = cell(1,NS);
  for k = 1:NS
    pmcounts{k} = zeros(NT,TT);
    pmcounts2{k} = zeros(NT,TT);
    pcounts{k} = zeros(NT,1);
    pmsem{k} = zeros(NT,TT);
  end
end
%*******
mcounts = zeros(NT,TT);   %response to stimulus
mcounts2 = zeros(NT,TT);
counts = zeros(NT,1);
msem = zeros(NT,TT);
for k = 1:SN
    svec = StimX(k,4:end);
    pvec = StimPX(k,4:end);
    z = find( svec ~= 0);   
    if ~isempty(z)
        if ( (k+DTA) > 0) && ((k+DTB)<=SN)
             spcnts = StimY((k+XX)',1)';
             for ik = 1:length(z)  
               it = z(ik);
               mcounts(it,:) = mcounts(it,:) + spcnts;
               mcounts2(it,:) = mcounts2(it,:) + spcnts .^ 2;
               counts(it) = counts(it) + 1;
               if (NS > 1)
                 sp = find( StimX(k,3) == spark );
                 pmcounts{sp}(it,:) = pmcounts{sp}(it,:) + spcnts;
                 pmcounts2{sp}(it,:) = pmcounts2{sp}(it,:) + spcnts .^ 2;
                 pcounts{sp}(it) = pcounts{sp}(it) + 1;  
               end
               %********
               di = svec(it);  %motion direction, 1 to ND
               if (di <= ND)
                 zmcounts{di}(it,:) = zmcounts{di}(it,:) + spcnts;
                 zmcounts2{di}(it,:) = zmcounts2{di}(it,:) + spcnts .^ 2;
                 zcounts{di}(it) = zcounts{di}(it) + 1;
               end
               %********
               di = pvec(it);  %motion speed, 1 to NP
               if (di <= NP)
                 zpcounts{di}(it,:) = zpcounts{di}(it,:) + spcnts;
                 zpcounts2{di}(it,:) = zpcounts2{di}(it,:) + spcnts .^ 2;
                 ppcounts{di}(it) = ppcounts{di}(it) + 1;
               end
               %**********
             end
        end
    end
    if (mod(k,floor(SN/10)) == 0)
       disp(sprintf('Processing %5.1f percent comp',(k*100/SN))); 
    end
end
for it = 1:NT
    if counts(it) > 0
       mcounts(it,:) = mcounts(it,:) / counts(it);  % mean
       msem(it,:) = sqrt(  ((mcounts2(it,:)/counts(it)) - mcounts(it,:).^2) / counts(it));  % 1 sem 
       mcounts(it,:) = mcounts(it,:) / SampRate;
       msem(it,:) = msem(it,:) / SampRate;
    else
       mcounts(it,:) = NaN(size(mcounts(it,:)));
       msem(it,:) = NaN(size(mcounts(it,:)));
    end
end
%******** now by motion dir
for di = 1:ND
 for it = 1:NT
    if zcounts{di}(it) > 0
       zmcounts{di}(it,:) = zmcounts{di}(it,:) / zcounts{di}(it);  % mean
       zmsem{di}(it,:) = sqrt(  ((zmcounts2{di}(it,:)/zcounts{di}(it)) - zmcounts{di}(it,:).^2) / zcounts{di}(it));  % 1 sem 
       zmcounts{di}(it,:) = zmcounts{di}(it,:) / SampRate;
       zmsem{di}(it,:) = zmsem{di}(it,:) / SampRate;
    else
       zmcounts{di}(it,:) = NaN(size(zmcounts{di}(it,:)));
       zmsem{di}(it,:) = NaN(size(zmcounts{di}(it,:)));
    end
 end
end
%******** now by motion speed
for di = 1:NP
 for it = 1:NT
    if ppcounts{di}(it) > 0
       zpcounts{di}(it,:) = zpcounts{di}(it,:) / ppcounts{di}(it);  % mean
       zpsem{di}(it,:) = sqrt(  ((zpcounts2{di}(it,:)/ppcounts{di}(it)) - zpcounts{di}(it,:).^2) / ppcounts{di}(it));  % 1 sem 
       zpcounts{di}(it,:) = zpcounts{di}(it,:) / SampRate;
       zpsem{di}(it,:) = zpsem{di}(it,:) / SampRate;
    else
       zpcounts{di}(it,:) = NaN(size(zpcounts{di}(it,:)));
       zpsem{di}(it,:) = NaN(size(zpcounts{di}(it,:)));
    end
 end
end
%****** and now by spareseness
if (NS > 1)
 for sp = 1:NS
  for it = 1:NT
    if pcounts{sp}(it) > 0
       pmcounts{sp}(it,:) = pmcounts{sp}(it,:) / pcounts{sp}(it);  % mean
       pmsem{sp}(it,:) = sqrt(  ((pmcounts2{sp}(it,:)/pcounts{sp}(it)) - pmcounts{sp}(it,:).^2) / pcounts{sp}(it));  % 1 sem 
       pmcounts{sp}(it,:) = pmcounts{sp}(it,:) / SampRate;
       pmsem{sp}(it,:) = pmsem{sp}(it,:) / SampRate;
    else
       pmcounts{sp}(it,:) = NaN(size(pmcounts{sp}(it,:)));
       pmsem{sp}(it,:) = NaN(size(pmcounts{sp}(it,:)));
    end
  end
 end
end

%****** compute significant time/space points for averaging
Gsig = 0.5;  % smooth in 3d kernel space to flag sig points
%*** rescale bonferoni correction *******
uu = zeros(Nx,Ny,TT);
uu(floor(Nx/2),floor(Ny/2),floor(TT/2)) = 1;
iuu = imgaussfilt3(uu,Gsig);
iuu = iuu/max(max(max(iuu)));  % put peak at 1.0
NPP = sum(sum(sum(iuu)));
bonfsig = -norminv(BONFP * NPP);
%********

  sigcounts = zeros(size(mcounts));
  %*** flag points that exceed bonfsig
  zbase = find( tXX < 30);  % use values before visual latency
  %********
  uu = nanmean( mcounts(:,zbase)' )';
  su = nanstd( mcounts(:,zbase)' )';
  NoiseFloor = su;  % use later to filter badly sampled locations
  %*** compute t-values and smooth them
  tvals = zeros(size(mcounts));
  for k = 1:size(mcounts,2)
      tvals(:,k) = ((mcounts(:,k)-uu) ./ su );
  end
  %**** now smooth the tvals ******
  ntvals = zeros(Nx,Ny,TT); %tvals;
  for k = 1:size(mcounts,2)
      ntvals(:,:,k) = reshape( squeeze( tvals(:,k)),Nx,Ny);
  end
  sntvals = imgaussfilt3(ntvals,Gsig);
  %*******
  ntvals = tvals;
  for k = 1:size(mcounts,2)
      ntvals(:,k) = reshape( squeeze(sntvals(:,:,k)),(Nx*Ny),1);
  end
  %***** now flag points outside those intervals
  for k = 1:size(mcounts,2)
    if (tXX(k) >= 30)  % enforce visual latency
       zit = k;
       zp = find( ntvals(:,zit) > bonfsig);
       sigcounts(zp,k) = 1;  % mark sig locations
       zp = find( ntvals(:,zit) < -bonfsig);
       sigcounts(zp,k) = -1;  % mark sig locations
    end
  end

%******** compute selectivity by motion, using positive sig points
%******** as the input samples for mean and std of firing
mou = [];
mostd = [];
siglist = find( sigcounts >= 1);  % list of sig points
for k = 1:ND
   mou(1,k) = nanmean( zmcounts{k}( siglist ) );
   mostd(1,k) = nanstd( zmcounts{k}( siglist ))/ sqrt(length(siglist));
end
%***** need to think here, but could look at each sig location and
%***** estimate a motion vector for it, then plot those as arrows or color

%******** compute selectivity by motion speed, using positive sig points
%******** as the input samples for mean and std of firing
pou = [];
postd = [];
siglist = find( sigcounts >= 1);  % list of sig points
for k = 1:NP
   pou(1,k) = nanmean( zpcounts{k}( siglist ) );
   postd(1,k) = nanstd( zpcounts{k}( siglist ))/ sqrt(length(siglist));
end

%******* Smooth motion kernels *********
if (1)
  SGsig = 1.0;
  imou = [mou mou mou];
  imostd = [mostd mostd mostd];
  simou = imgaussfilt(imou,SGsig);
  simostd = imgaussfilt(imostd,SGsig);
  mou = simou((ND+1):(2*ND));
  mostd = simostd((ND+1):(2*ND));
end
%*********


%******* compute cumulative RF from all sig timepoints
zt = 1:(2-DTA+KN-1);  % set shown as RF plot (only test those)
numcounts = sum( sigcounts(:,zt) > 0 );
usig = mean(numcounts);
z = find( numcounts > 0);
thresh = 0;
if (length(z) > 2)
    thresh = mean(numcounts(z));
end
sigt = find( numcounts > thresh );  % take peak, or any if less than 2 sig

%**** throw out spatial locations with less that ThrowSig standard deviations
%**** of noise floor (not enough samples at some spatial location)
ThrowSig = 2;
ZNoise = ones(Nx,Ny);
if (ThrowSig)
    nf = flipud( reshape(NoiseFloor',Nx,Ny)' );   
    unf = nanmean(nanmean(nf));
    sunf = nanstd(nanstd(nf));
    zz = find( nf > (unf + 3*sunf));
    ZNoise(zz) = 0;
end

%********* Build a smoothing kernel that considers the NoiseFloor
%******* due to different sampling 
ZSmooth = zeros(NT,NT);  % smoothing with non-uniform sizes
unf = nanmedian(NoiseFloor);
SmoothSig = 0.5 * (NoiseFloor / unf);  % smooth in ratio so median is 0.5
for ii = 1:NT
    xi = 1 + mod((ii-1), Nx);
    yi = 1 + floor( (ii-1) / Nx);
    sigo = SmoothSig(ii);
    if (sigo == 0) || isnan(sigo)
        sigo = nanmax(SmoothSig);  % used largest to interpolate point
    end
    for jj = 1:NT
       xj = 1 + mod((jj-1), Nx);
       yj = 1 + floor( (jj-1) / Nx);
       dist = ((xi-xj)^2 + (yi-yj)^2)/(sigo^2);
       if (dist < 10)
            ZSmooth(ii,jj) = exp( -0.5*dist );
       else
            ZSmooth(ii,jj) = 0;
       end
    end
    ZSmooth(ii,:) = ZSmooth(ii,:) / sum(ZSmooth(ii,:));
end

%**** compute average cumalitive RF from sig time-points
if (length(sigt) > 1) 
   meanrf = flipud( reshape(nanmean( mcounts(:,sigt)')',Nx,Ny)' );
else
   %***** pick a default time bin if none are sig ***********
   sigt = find( (tXX >= 50) & (tXX < 100));
   meanrf = flipud( reshape(nanmean(mcounts(:,sigt)')',Nx,Ny)' ); 
end

%****** replace any NaN points (low sampling) with mean
moo = reshape( flipud(meanrf)', 1, NT);   % inverse transform back
umoo = nanmean(nanmean(moo));
zz = find( isnan(moo) );
moo(zz) = umoo;
%****** apply smoothing operation
zmm = ZSmooth * moo';
meanrf = flipud( reshape(zmm,Nx,Ny)' );

mino = nanmin(nanmin(meanrf));
maxo = nanmax(nanmax(meanrf));
%*** compute means as a func of sparseness ******
if (NS > 1)
  sparserf = cell(1,NS);
  for k = 1:NS
     if (length(sigt) > 1)
        sparserf{k} = flipud( reshape(nanmean( pmcounts{k}(:,sigt)')',Nx,Ny)' );
        zz = find(ZNoise == 0);
        uu = nanmean(nanmean(sparserf{k}));
        sparserf{k}(zz) = uu; 
     else
        sparserf{k} = flipud( reshape((pmcounts{k}(:,sigt)')',Nx,Ny)' );
        zz = find(ZNoise == 0);
        uu = nanmean(nanmean(sparserf{k}));
        sparserf{k}(zz) = uu; 
     end
     %****** do same as above, for sparseRF to smooth
     moo = reshape( flipud(sparserf{k})', 1, NT);   % inverse transform back
     umoo = nanmean(nanmean(moo));
     zz = find( isnan(moo) );
     moo(zz) = umoo;
     %****** apply smoothing operation
     zmm = ZSmooth * moo';
     sparserf{k} = flipud( reshape(zmm,Nx,Ny)' );
     %********
     mno = nanmin(nanmin(meanrf));
     mxo = nanmax(nanmax(meanrf));
     mino = min(mino,mno);
     maxo = max(maxo,mxo);
  end
end

%****** store necessary information in rfinfo struct to plot
rfinfo.RFPlotMino = mino;
rfinfo.RFPlotMaxo = maxo;
rfinfo.ZSmooth = ZSmooth;
rfinfo.SmoothSig = SmoothSig;
rfinfo.NoiseFloor = NoiseFloor;
rfinfo.KN = KN;
rfinfo.Nx = Nx;
rfinfo.Ny = Ny;
rfinfo.TT = TT;
rfinfo.tXX = tXX;
rfinfo.NT = NT;
rfinfo.DTA = DTA;
rfinfo.DTB = DTB;
rfinfo.mcounts = mcounts;
rfinfo.BONFP = BONFP;
rfinfo.sigcounts = sigcounts;
rfinfo.mou = mou;
rfinfo.mostd = mostd;
rfinfo.pou = pou;
rfinfo.postd = postd;
rfinfo.Zx = Zx;
rfinfo.Zy = Zy;
rfinfo.sigt = sigt;
rfinfo.meanrf = meanrf;
rfinfo.NS = NS;
if (NS > 1)
    rfinfo.spark = spark;
    rfinfo.sparserf = sparserf;
end
rfinfo.NP = NP;

%********* Plotting routines below ***************
if (0)
  hf = figure;
  set(hf,'position',[150 50 1200 900]);
  subplot('position',[0.1 0.75 0.25 0.20]);
  plot(tSXX,mvec(SACN+1,:),'k-'); hold on;
  plot(tSXX,mvec(SACN+1,:) + (2*mvec2(SACN+1,:)),'k-'); hold on;
  plot(tSXX,mvec(SACN+1,:) - (2*mvec2(SACN+1,:)),'k-'); hold on;
  axis tight;
  V = axis;
  plot([0,0],[V(3),V(4)],'k-');
  plot([POSTPERIOD(1),POSTPERIOD(1)],[V(3),V(4)],'b-');
  plot([POSTPERIOD(2),POSTPERIOD(2)],[V(3),V(4)],'b-');
  plot([DURPERIOD(1),DURPERIOD(1)],[V(3),V(4)],'m-');
  plot([DURPERIOD(2),DURPERIOD(2)],[V(3),V(4)],'m-');
  plot([PREPERIOD(1),PREPERIOD(1)],[V(3),V(4)],'r-');
  plot([PREPERIOD(2),PREPERIOD(2)],[V(3),V(4)],'r-');
  plot([tSXX(1),tSXX(end)],[sacbase,sacbase],'k-');
  plot([tSXX(1),tSXX(end)],[sacbase+(2*sacbasestd),sacbase+(2*sacbasestd)],'k-');
  plot([tSXX(1),tSXX(end)],[sacbase-(2*sacbasestd),sacbase-(2*sacbasestd)],'k-'); 
  xlabel('Time (ms)');
  ylabel('Firing Rate');
  title('Average Saccade Modulation');
  subplot('position',[0.4 0.75 0.25 0.20]);
  for k = 1:SACN
    plot(tSXX,mvec(k,:),'k-'); hold on;
  end
  axis tight;
  V = axis;
  plot([0,0],[V(3),V(4)],'k-');
  plot([POSTPERIOD(1),POSTPERIOD(1)],[V(3),V(4)],'b-');
  plot([POSTPERIOD(2),POSTPERIOD(2)],[V(3),V(4)],'b-');
  plot([DURPERIOD(1),DURPERIOD(1)],[V(3),V(4)],'m-');
  plot([DURPERIOD(2),DURPERIOD(2)],[V(3),V(4)],'m-');
  plot([PREPERIOD(1),PREPERIOD(1)],[V(3),V(4)],'r-');
  plot([PREPERIOD(2),PREPERIOD(2)],[V(3),V(4)],'r-');
  xlabel('Time (ms)');
  ylabel('Firing Rate');
  title('Average Saccade Modulation');
  subplot('position',[0.7 0.75 0.25 0.20]);
  xx = (0:(ND+6))*(360/ND);
  yy = [1:ND,1:(size(xx,2)-ND)];
  plot(xx,postcurve(yy),'b-'); hold on;
  plot(xx,postcurve(yy)+(2*postcurve_std(yy)),'b-'); hold on;
  plot(xx,postcurve(yy)-(2*postcurve_std(yy)),'b-'); hold on;
  plot(xx,durcurve(yy),'m-'); hold on;
  plot(xx,durcurve(yy)+(2*durcurve_std(yy)),'m-'); hold on;
  plot(xx,durcurve(yy)-(2*durcurve_std(yy)),'m-'); hold on;
  plot(xx,precurve(yy),'r-'); hold on;
  plot(xx,precurve(yy)+(2*precurve_std(yy)),'r-'); hold on;
  plot(xx,precurve(yy)-(2*precurve_std(yy)),'r-'); hold on;
  axis tight;
  plot([xx(1),xx(end)],[sacbase,sacbase],'k-');
  plot([xx(1),xx(end)],[sacbase+(2*sacbasestd),sacbase+(2*sacbasestd)],'k-');
  plot([xx(1),xx(end)],[sacbase-(2*sacbasestd),sacbase-(2*sacbasestd)],'k-'); 
  xlabel('Direction (degs)');
  ylabel('Saccade Rate');
  title('Saccade Mod by Direction');
end

%**** plot RF across time frames, and show significant points
if (0)
    dx = (0.9/KN);
    dxs = (0.9/(KN+1));
    mino = min(min(mcounts));
    maxo = max(max(mcounts));
    for it = 1:(KN-1)
       fx = 0.075 + ((it-1)*dx);
       subplot('position',[fx 0.55 dxs dxs]);
       ito = (2-DTA)+it;
       svec = flipud( reshape(squeeze(mcounts(:,ito)),Nx,Ny)' );
       imagesc(Zx,Zy,svec,[mino maxo]); hold on;
       plot([-11,11],[0,0],'k-');
       plot([0,0],[-11,11],'k-');
       axis off;
       h = title(sprintf('%4.1f',-tXX(ito)));
       %********* mark sig locations if any
       subplot('position',[fx 0.45 dxs dxs]);
       svec = flipud( reshape(squeeze(sigcounts(:,ito)),Nx,Ny)' );
       imagesc(Zx,Zy,svec,[-1 1]); hold on;
       plot([-11,11],[0,0],'r-');
       plot([0,0],[-11,11],'r-');
       axis off;
       %***********
    end
    fx = 0.075 + ((KN-1)*dx);
    subplot('position',[fx 0.55 dxs dxs]);
    imagesc(Zx,Zy,ones(size(svec))*mino,[mino maxo]); hold on;
    axis off;
    h = colorbar;
    
    %***********
    subplot('position',[0.05 0.15 0.2 0.2]);
    imagesc(Zx,Zy,meanrf,[mino maxo]); hold on;
    plot([-11,11],[0,0],'k-');
    plot([0,0],[-11,11],'k-');
    axis off;
    title('Mean RF');
    %************   
   
    %****** plot RFs by sparseness
    if (NS > 1)
        dx = (0.4/(NS/2));
        dxs = (0.4/((NS/2)+1));
        for sp = 1:NS
          iy = 1-mod((sp-1),2);
          ix = floor((sp-1)/2);
          fx = 0.30 + (ix*dx);
          fy = 0.05 + (iy*dx);
          subplot('position',[fx fy dxs dxs]);
          imagesc(Zx,Zy,sparserf{sp},[mino maxo]); hold on;
          plot([-11,11],[0,0],'k-');
          plot([0,0],[-11,11],'k-');
          axis off;
          title(sprintf('Sparse=%d',spark(sp)));
        end
    end
    
    %********** plot motion selectivity over sig points
    subplot('position',[0.7 0.10 0.25 0.3]);
    xx = (0:(ND+6))*(360/ND);
    yy = [1:ND,1:(size(xx,2)-ND)];
    plot(xx,mou(yy),'k-'); hold on;
    plot(xx,mou(yy)+(2*mostd(yy)),'k-'); hold on;
    plot(xx,mou(yy)-(2*mostd(yy)),'k-'); hold on;
    plot([xx(1),xx(end)],[sacbase,sacbase],'k-');
    plot([xx(1),xx(end)],[sacbase+(2*sacbasestd),sacbase+(2*sacbasestd)],'k-');
    plot([xx(1),xx(end)],[sacbase-(2*sacbasestd),sacbase-(2*sacbasestd)],'k-'); 
    %***** replot single lines from sac modulation for comparison
    plot(xx,postcurve(yy),'b-'); hold on;
    plot(xx,durcurve(yy),'m-'); hold on;
    plot(xx,precurve(yy),'r-'); hold on;
    %**********
    axis tight;
    xlabel('Direction (degs)');
    ylabel('Firing (hz)');
    title('Motion selectivity');
    %***************
end

return;
