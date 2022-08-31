function [StimX,SacX,StimY,StimPX] = StimMatrix_ForageMotionSpatialSacKernel(Exp,SP,GRID)
% function StimMatrix_ForageSpatialKernel(Exp)
%   input: takes Exp struct as the input
%          SP - spike channels to analyze
%          GRID with fields:
%             GRID.box = [cent x, cent y, width, height]
%             GRID.div = size of spatial pixel
%
%  Exp fields:
%    D: {100x1 cell}  - data per trial
%    S: [1x1 struct]  - rig settings struct
%   sp: [1x1 struct]  - spikes from spike sorting
%   cp: (will be a 60hz filtered LFP at 1000 hz)
%   ey: (will be a struct for eye position)
% Outputs:
%            note, StimX,StimY are relative to the eye position (eye corrected)
%   StimX: [Nframes,3+NK] - fields, matlab time, ephys time, NK = size of grid
%   SacX:  [Nframes,3] - marked on onset, amp, int of sac dir, sac dir
%   StimY: [Nframes,Nsp] - spike counts per video frame, for all Nsp units
%   StimPX: [Nframes,3+NK] - same as StimX, but values indicate speed,
%                            rather than the motion direction
%*************
%
% written by JM

% sp = Exp.sp{SPclust};
StimX = [];
StimPX = [];
SacX = [];
StimY = [];

%**** transform some parameters of the grid
Nx = floor(GRID.box(3)/GRID.div);
Ny = floor(GRID.box(4)/GRID.div);
NT = (Nx * Ny);
BaseX = GRID.box(1) - (GRID.box(3)/2);
BaseY = GRID.box(2) - (GRID.box(4)/2);
Div = GRID.div;
if isempty(SP)
    Nsp = length(Exp.sp);
    XSP = 1:Nsp;
else
    Nsp = length(SP);
    XSP = SP;
end
%***********************

%**** go through trials, building a stimulus representations
%**** in the coordinate frame of spike times ***************
%**** at this time, no consideration of saccades or probe
%**** stimulus timing is considered (later)
%***
%***** first scroll trials to compute size of stim array in time
Nstim = 0;
disp('Testing for trials with spatial mapping');
for zk = 1:size(Exp.D,1)
    if (length(Exp.D{zk}.PR.name) >= 6)
        if strcmp(Exp.D{zk}.PR.name(1:6),'Forage')
            if (Exp.D{zk}.PR.noisetype == 5) % Motion at Spatial
                fprintf('Including trial %d\n',zk);
                Nstim = Nstim + size(Exp.D{zk}.PR.NoiseHistory,1);
            end
        end
    end
end
fprintf('Collected %d frames\n',Nstim);
if (Nstim == 0)
    return;
end

%****** then allocate space and build structure
StimX = zeros(Nstim,4+NT-1);  % mat_time, ephy_time, stimulus, noisecap
StimPX = zeros(Nstim,4+NT-1);  % mat_time, ephy_time, stimulus, noisecap
SacX = zeros(Nstim,3);  % amplitude (0 for none), direction (1-16), direction (degs)
StimY = zeros(Nstim,Nsp);  % spike counts per frame epoch
StimK = 0;
for zk = 1:size(Exp.D,1)
    if (length(Exp.D{zk}.PR.name) >= 6)
        if (strcmp(Exp.D{zk}.PR.name(1:6),'Forage'))  % keep in mind, other trials could be image backgrounds
            if (Exp.D{zk}.PR.noisetype == 5) % Spatial
                %******************
                noisecap = Exp.D{zk}.PR.noiseNum;  % default
                if isfield(Exp.D{zk}.PR,'noiseCap')
                    noisecap = Exp.D{zk}.PR.noiseCap;
                end
                %*************
                matstart = Exp.D{zk}.eyeData(1,6);  % register frame start clock
                matend = Exp.D{zk}.eyeData(end,6);  % register frame end clock
                epystart = Exp.D{zk}.START_EPHYS;   % matching strobe time ephys start
                epyend = Exp.D{zk}.END_EPHYS;       % matching strobe time ephys end
                eyesmo = Exp.D{zk}.eyeSmo;          % recorded eye position
                slist = Exp.D{zk}.slist;            % saccade times relative to data
                %******** build stimulus matrix in spike time coordinates
                timvec = Exp.D{zk}.PR.NoiseHistory(:,1);  % delayed by 8 ms, fix in stim
                epyvec = (timvec-matstart)/(matend-matstart);
                epyvec = epystart + epyvec * (epyend - epystart);
                %***************************
                ia = StimK+1;
                ib = StimK+size(timvec,1);
                StimX(ia:ib,1) = timvec;
                StimX(ia:ib,2) = epyvec;
                StimX(ia:ib,3) = noisecap;
                %****
                StimPX(ia:ib,1) = timvec;
                StimPX(ia:ib,2) = epyvec;
                StimPX(ia:ib,3) = noisecap;
                %******* loop through to compute spatial positions
                KT = size(timvec,1);
                svec = zeros(KT,NT);
                pvec = zeros(KT,NT);
                %****
                for k = 1:KT
                    %******* determine if saccade onset in the current frame
                    if (k < KT)
                        framet = timvec(k) - matstart;
                        framet2 = timvec(k+1) - matstart;
                        %****** saccade onset in this frame?
                        if ~isempty(slist)
                            z = find ( (slist(:,1) >= framet) & (slist(:,1) < framet2) );
                            if ~isempty(z)  % found a saccade onset
                                stasac = slist(z(1),4); % integer start time of saccade
                                endsac = slist(z(1),5); % integer end time of saccade
                                stx = eyesmo(stasac,2);
                                sty = eyesmo(stasac,3);
                                etx = eyesmo(endsac,2);
                                ety = eyesmo(endsac,3);
                                sacvec = [etx-stx,ety-sty];
                                ampo = norm(sacvec);
                                ango = angle(complex(sacvec(1),sacvec(2))) * (360/(2*pi));
                                %*** break into 16 directions on circle (same as
                                %*** break down for motion stimulus
                                if (ango < -11.25)
                                    ango = ango + 360;
                                end
                                if (ango > 348.75)
                                    ango = ango - 360;
                                end
                                sacdir = 1+floor((ango+11.25)/22.5);  % scales 1 to 16
                                %****** store saccade
                                SacX(StimK+k,1) = ampo;
                                SacX(StimK+k,2) = sacdir;
                                SacX(StimK+k,3) = ango;
                            end
                        end
                    end
                    %***************
                    framet = timvec(k) - matstart;
                    if isempty(eyesmo)
                        z = [];
                    else
                        z = find( eyesmo(:,1) >= framet );
                    end
                    if ~isempty(z)
                        ex = eyesmo(z(1),2);
                        ey = eyesmo(z(1),3);

                        for sk = 1:Exp.D{zk}.PR.noiseNum
                            sx = Exp.D{zk}.PR.NoiseHistory(k,2+((sk-1)*4));  % x pos
                            sy = Exp.D{zk}.PR.NoiseHistory(k,3+((sk-1)*4));  % y pos
                            sdir = Exp.D{zk}.PR.NoiseHistory(k,4+((sk-1)*4));  % motion direction (angle)
                            slife = Exp.D{zk}.PR.NoiseHistory(k,5+((sk-1)*4));  % lifetime 1-5, 3 middle
                            if ~isfield(Exp.D{zk}.PR,'SpeedHistory')
                                if ~isfield(Exp.D{zk}.P,'snoisespeed')
                                    spd = 15;
                                else
                                    spd = Exp.D{zk}.P.snoisespeed;
                                end
                            else
                                spd = Exp.D{zk}.PR.SpeedHistory(k,1+sk);  % dot speed (angle)
                            end

                            if (slife == 1)  % time-lock only to the first flash (valid timing)
                                %*** find future position of stimulus at its
                                %*** mid-point, but time-lock from its onset
                                kk = k + 2;  % frame 3 is centered
                                if (kk <= KT)
                                    framet = timvec(kk) - matstart;
                                    z = find( eyesmo(:,1) >= framet );
                                    if ~isempty(z)
                                        ex = eyesmo(z(1),2);
                                        ey = eyesmo(z(1),3);
                                    end
                                    sx = Exp.D{zk}.PR.NoiseHistory(kk,2+((sk-1)*4));  % x pos
                                    sy = Exp.D{zk}.PR.NoiseHistory(kk,3+((sk-1)*4));  % y pos
                                end

                                %********* determine grid pixel and mark with val
                                sxx = sx - ex;  % recentered by eye pos
                                syy = sy - ey;
                                %****
                                ix = 1 + floor((sxx - BaseX - 0)/Div);
                                iy = 1 + floor((syy - BaseY - 0)/Div);
                                % ix = 1 + floor((sxx - BaseX -(Div/2))/Div);
                                % iy = 1 + floor((syy - BaseY -(Div/2))/Div);
                                if ( (ix >= 1) && (ix <= Nx) && (iy >= 1) && (iy <= Ny) )
                                    vox = (iy-1)*Nx + ix;
                                    svec(k,vox) = sdir; % can't deal with super-position anymore, svec(k,vox) + sdir;
                                    pvec(k,vox) = spd;
                                    %*** how to convert int speed to real one
                                    % spd = randi(o.P.snoisespds);
                                    % noisespeed = o.P.snoisebase^spd;
                                    % o.hNoise(7,kk) = spd;
                                end
                            end
                            %*******************************************
                        end
                    end
                end
                %****
                StimX(ia:ib,4:(NT+3)) = svec;
                StimPX(ia:ib,4:(NT+3)) = pvec;
                %****** now compute counts via frame
                for skk = 1:Nsp
                    sk = XSP(skk);
                    if ~isempty(Exp.sp{sk}) && ~isempty(Exp.sp{sk}.st)
                        for it = 2:size(epyvec,1)
                            zz = find( (Exp.sp{sk}.st >= epyvec(it-1)) & (Exp.sp{sk}.st < epyvec(it) ) );
                            StimY(ia+(it-1),skk) = size(zz,1);
                        end
                    end
                end
                StimK = StimK + size(timvec,1);
                %*************************
            end
        end
    end
    if (zk == 1) || (mod(zk,10)==0)
        fprintf('Processing trial %d for noise history\n',zk);
    end
end
StimX = StimX(1:StimK,:);
StimPX = StimPX(1:StimK,:);
SacX = SacX(1:StimK,:);
StimY = StimY(1:StimK,:);
fprintf('Processing finished\n');
%*********************************

return;
