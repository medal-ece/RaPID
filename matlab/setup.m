%% ======================= USER SETTINGS =====================================
clear; clc;

csvFile = 'GainPlot_1x1.csv';   % Element pattern file

freq    = 2.4e9;                % Hz
c       = 3e8;                  % m/s
lambda  = c/freq;
k       = 2*pi/lambda;
dx = 0.5*lambda;                % spacing in x between element columns
dy = 0.5*lambda;                % spacing in y between element rows
M  = 4;                         % elements in x
N  = 4;                         % elements in y

phi_step = 1;                   % deg
phi_off  = phi_step/2;          % half-step offset (HFSS-like mid-bin)
phi_deg_range   = (-180+phi_off):phi_step:(180-phi_off);
theta_deg_range = 0:1:180;
cap_thetaMaxDeg = 70;

Beta_deg = [0, 0, 100, 100; 0, 0, 100, 100; 0, 0, 100, 100; 0, 0, 100, 100];
%%% 100 main-lobe angles, evenly over -170<phi<170 and 0<theta<180 %%%
%%% Normal-distribution perturbation around the base dx,dy (in degrees) %%%
num_batches = 25;
batch_size    = 1;      % 50 datasets per batch (per main-lobe angle)
phase_sigma_deg = 5;   % std dev of phase noise; adjust as desired

normalize_to_HFSS = false;
normalize_to_0dB = true;     % show shapes only
normalizeAF = false;         % true -> AF normalized by MN (shape only)
useCenteredOrigin = true;   % true -> center array about (0,0)

%% ==================== Load element pattern (power) ========================
[PHI_deg, THETA_deg] = meshgrid(phi_deg_range, theta_deg_range);
TH = deg2rad(THETA_deg);
PH = deg2rad(PHI_deg);
rawTbl = [];  try, rawTbl = readtable(csvFile); catch, end
if ~isempty(rawTbl) && width(rawTbl) >= 3
    % scattered [?, ?, gain_dBi] – auto-detect column order
    c1 = rawTbl{:,1}; c2 = rawTbl{:,2}; g3 = rawTbl{:,3};
    looksTheta = @(v) min(v)>=0 && max(v)<=180;
    looksPhi   = @(v) (min(v)>=-180 && max(v)<=180) || (min(v)>=0 && max(v)<=360);
    if looksTheta(c1) && looksPhi(c2)
        theta_s=c1; phi_s=c2;
    elseif looksPhi(c1) && looksTheta(c2)
        phi_s=c1; theta_s=c2;
    else
        phi_s=c1; theta_s=c2;
    end
    phi_s = mod(phi_s+180,360)-180;
    th_ext = [theta_s; theta_s; theta_s];
    ph_ext = [phi_s-360; phi_s; phi_s+360];
    g_ext  = [g3; g3; g3];
    F = scatteredInterpolant(th_ext, ph_ext, g_ext, 'natural','nearest');
    elem_gain_dBi = F(THETA_deg, PHI_deg);
else
    A = readmatrix(csvFile); % grid with headers
    theta_header = A(2:end,1);
    phi_header   = mod(A(1,2:end)+180,360)-180;
    gain_grid    = A(2:end,2:end);
    [PHI_src, THETA_src] = meshgrid(phi_header, theta_header);
    G = griddedInterpolant(THETA_src, PHI_src, gain_grid, 'linear','nearest');
    elem_gain_dBi = G(THETA_deg, PHI_deg);
end

elem_gain_lin = 10.^(elem_gain_dBi/10);  % linear power element pattern

