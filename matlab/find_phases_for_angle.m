%% ==================== Helper: find phases for target main lobe (SEARCH) ===
function [dx_opt, dy_opt, G_opt] = find_phases_for_angle( ...
    M,N, theta0_deg,phi0_deg, elem_gain_lin,TH,PH,dx,dy,k, ...
    normalize_to_0dB, phi_deg_range,theta_deg_range, cap_thetaMaxDeg)

    % ---- Good initial guess from closed-form steering law (very important) ----
    theta0 = deg2rad(theta0_deg);
    phi0   = deg2rad(phi0_deg);
    dx0 = -rad2deg(k*dx*sin(theta0)*cos(phi0));
    dy0 = -rad2deg(k*dy*sin(theta0)*sin(phi0));

    % Wrap to a consistent range to avoid huge equivalent solutions
    dx0 = wrap180(dx0);
    dy0 = wrap180(dy0);

    % ---- Coarse local search around (dx0,dy0) to avoid fminsearch stagnation ----
    coarse_span = 120;   % deg
    coarse_step = 10;   % deg
    dgrid = -coarse_span:coarse_step:coarse_span;

    bestJ = Inf;
    bestx = [dx0 dy0];

    for ddx = dgrid
        for ddy = dgrid
            xt = [wrap180(dx0+ddx), wrap180(dy0+ddy)];
            Jt = totalpattern_cost(xt, M,N, theta0_deg,phi0_deg, ...
                elem_gain_lin,TH,PH,dx,dy,k, phi_deg_range,theta_deg_range, cap_thetaMaxDeg);
            if Jt < bestJ
                bestJ = Jt;
                bestx = xt;
            end
        end
    end

    % ---- Refine with fminsearch (uses total-pattern objective, not peak index only) ----
    costFun = @(x) totalpattern_cost(x, M,N, theta0_deg,phi0_deg, ...
        elem_gain_lin,TH,PH,dx,dy,k, phi_deg_range,theta_deg_range, cap_thetaMaxDeg);

    options = optimset('Display','off', ...
                       'MaxIter', 250, 'MaxFunEvals', 800, ...
                       'TolX', 1e-3, 'TolFun', 1e-6);

    x_opt = fminsearch(costFun, bestx, options);

    dx_opt = wrap180(x_opt(1));
    dy_opt = wrap180(x_opt(2));

    % Pattern for output (use your normalization setting for dataset)
    G_opt = simulate_array(M,N,dx_opt,dy_opt, ...
        elem_gain_lin,TH,PH,dx,dy,k,normalize_to_0dB);
end

function J = totalpattern_cost(x, M,N, theta0_deg,phi0_deg, ...
    elem_gain_lin,TH,PH,dx,dy,k, phi_deg_range,theta_deg_range, cap_thetaMaxDeg)

    dx_phase = wrap180(x(1));
    dy_phase = wrap180(x(2));

    % Total pattern WITHOUT normalization
    G_dB = simulate_array(M,N,dx_phase,dy_phase, elem_gain_lin,TH,PH,dx,dy,k,false);

    % ---------------- Fix A: ROI only ----------------
    roi_th = (theta_deg_range >= 0) & (theta_deg_range <= cap_thetaMaxDeg);
    theta_roi = theta_deg_range(roi_th);
    G_roi = G_dB(roi_th, :);

    % ---------------- Fix C: window around target ----------------
    thetaWin = 8;  phiWin = 8;      % tune
    win_th = abs(theta_roi - theta0_deg) <= thetaWin;

    dphi_vec = mod(phi_deg_range - phi0_deg + 180, 360) - 180;
    win_ph = abs(dphi_vec) <= phiWin;

    % Peak inside window (what we WANT to be the main beam)
    G_win = G_roi(win_th, win_ph);
    [Gwin_max, idxw] = max(G_win(:));
    [iw, jw] = ind2sub(size(G_win), idxw);

    ith_list = find(win_th);
    iph_list = find(win_ph);
    theta_p = theta_roi(ith_list(iw));
    phi_p   = phi_deg_range(iph_list(jw));

    % Peak outside window (competition)
    G_out = G_roi;
    G_out(win_th, win_ph) = -Inf;
    Gout_max = max(G_out(:));

    % Direction error (secondary)
    dtheta = theta_p - theta0_deg;
    dphi   = mod(phi_p - phi0_deg + 180, 360) - 180;

    % ---------------- Improved objective ----------------
    % (1) enforce dominance with a margin
    margin_dB = 1.0;   % require window peak >= outside peak + 1 dB
    dom_violation = max(0, (Gout_max + margin_dB) - Gwin_max);

    % (2) encourage the window peak to be high (prevents "both low" cheating)
    % Choose a mild weight; this term pushes toward stronger beams.
    w_gain = 0.02;       % tune 0.005~0.1
    gain_term = -Gwin_max; % maximize Gwin_max (since minimizing J)

    % (3) keep the peak near the target inside the window
    w_dir = 0.05;

    J = dom_violation^2 + w_dir*(dtheta^2 + dphi^2) + w_gain*gain_term;
end


function y = wrap180(x)
    y = mod(x + 180, 360) - 180;
end


