%% ==================== DATASET GENERATION SETUP ============================
setup;

% Angles (phi0,theta0) evenly distributed inside open ranges
[theta0_list, phi0_list, seeds_xyz] = cap_hex_centers(num_batches, cap_thetaMaxDeg, 12, 40000);

% Flattened angle grids (for saving patterns)
theta_flat = THETA_deg(:);               % (numAngles x 1)
phi_flat   = PHI_deg(:);                 % (numAngles x 1)
num_angles = numel(theta_flat);

% Output folder
outFolder = 'datasets_4x4';
if ~exist(outFolder,'dir'), mkdir(outFolder); end

rng(1); % reproducible randomness

% Store base patterns and base phases for verification
G_base_all = zeros(length(theta_deg_range), length(phi_deg_range), num_batches);
dx_base_all = zeros(num_batches,1);
dy_base_all = zeros(num_batches,1);

%% ==================== MAIN LOOP: 100 BATCHES =============================
for b = 1:num_batches
    phi0   = phi0_list(b);
    theta0 = theta0_list(b);

    % 1) Find base dx_phase, dy_phase for this main-lobe direction
    [dx_base, dy_base, G_base_dB] = find_phases_for_angle( ...
        M,N, theta0,phi0, elem_gain_lin,TH,PH,dx,dy,k, ...
        normalize_to_0dB, phi_deg_range,theta_deg_range,cap_thetaMaxDeg);
    
    % Store base pattern and phases for later verification plots
    G_base_all(:,:,b) = G_base_dB;
    dx_base_all(b)    = dx_base;
    dy_base_all(b)    = dy_base;

    % 2) Generate 50 realizations around the base phases
    dx_samples = dx_base + phase_sigma_deg*randn(batch_size,1);
    dy_samples = dy_base + phase_sigma_deg*randn(batch_size,1);

    % pattern_all: each column = one realization (flattened over (theta,phi))
    pattern_all = zeros(num_angles, batch_size);

    for s = 1:batch_size
        G_dB = simulate_array( ...
            M,N, dx_samples(s),dy_samples(s), ...
            elem_gain_lin,TH,PH,dx,dy,k, normalize_to_0dB);

        % P = 10.^(G_dB/10);
        % P = P ./ max(P(:));      % normalize to max = 1
        % pattern_all(:,s) = P(:); % store as column
        % G_dB is already normalized (max ~0 dB if normalize_to_0dB = true)
        pattern_all(:,s) = G_dB(:);   % store in dB
    end

    % 3) Build cell array for CSV
    num_cols = 2 + batch_size;

    % Header row: column names
    header = [{'theta_deg','phi_deg'}, ...
        arrayfun(@(s) sprintf('G_sample_%02d_dB',s), 1:batch_size, 'UniformOutput', false)];

    % Meta row with target angles (theta0, phi0)
    meta_row = cell(1,num_cols);
    meta_row{1} = 'target_theta_deg';
    meta_row{2} = theta0;
    if batch_size >= 2
        meta_row{3} = 'target_phi_deg';
        meta_row{4} = phi0;
    end

    % Rows with dx and dy phase for each of the 50 samples
    dx_row = [{'dx_phase_deg',''}, num2cell(dx_samples.')];
    dy_row = [{'dy_phase_deg',''}, num2cell(dy_samples.')];

    % Data rows: [theta_flat, phi_flat, P_sample_01..P_sample_50]
    data_rows = [ num2cell(theta_flat), num2cell(phi_flat), num2cell(pattern_all) ];

    C = [header; meta_row; dx_row; dy_row; data_rows];

    % File name encodes batch index and main-lobe direction
    fname = sprintf('batch_%03d_theta_%06.2f_phi_%+07.2f.csv', ...
                    b, theta0, phi0);
    writecell(C, fullfile(outFolder,fname));

    fprintf('Saved batch %3d/%3d: theta0 = %6.2f deg, phi0 = %7.2f deg, dx = %7.3f deg, dy = %7.3f deg\n', ...
        b, num_batches, theta0, phi0, dx_base, dy_base);
end

%% ==================== VERIFICATION: main-lobe directions ==================
roi_th = (theta_deg_range >= 0) & (theta_deg_range <= cap_thetaMaxDeg);
theta_roi = theta_deg_range(roi_th);

for b = 1:num_batches
    G_b = G_base_all(:,:,b);
    G_roi = G_b(roi_th, :);

    [~, idx] = max(G_roi(:));
    [ith_roi, iph] = ind2sub(size(G_roi), idx);

    theta_peak_list(b) = theta_roi(ith_roi);
    phi_peak_list(b)   = phi_deg_range(iph);
end

% theta_peak_list = zeros(num_batches,1);
% phi_peak_list   = zeros(num_batches,1);
% 
% for b = 1:num_batches
%     G_b = G_base_all(:,:,b);
%     [~, idx_max] = max(G_b(:));
%     [ith, iph]   = ind2sub(size(G_b), idx_max);
%     theta_peak_list(b) = theta_deg_range(ith);
%     phi_peak_list(b)   = phi_deg_range(iph);
% end
%% ==================== 2D PLOT: target vs achieved (theta,phi) =============
figure('Color','w'); hold on; grid on;

scatter(phi0_list,      theta0_list,      50, 'r', 'filled', ...
    'DisplayName','Target (\phi_0,\theta_0)');
scatter(phi_peak_list,  theta_peak_list,  35, 'bo', ...
    'DisplayName','Achieved peak');

xlabel('\phi (deg)');
ylabel('\theta (deg)');
title('Target vs. achieved main-lobe directions');
legend('Location','best');

%% ==================== 2D PLOT: Equal-area hemisphere centers =============
% Plot the centers in Lambert equal-area projection
% k = sqrt(2./(1 + seeds_xyz(:,3)));   % z = seeds_xyz(:,3)
% xp = k .* seeds_xyz(:,1);
% yp = k .* seeds_xyz(:,2);
% 
% figure('Color','w');
% scatter(xp, yp, 40, 'filled'); grid on; axis equal;
% title('~Equal-area hemisphere centers (Lambert azimuthal equal-area projection)');
% xlabel('x_{proj}'); ylabel('y_{proj}');

% Monte-Carlo Voronoi coloring in equal-area projection
Ns = 50000;
u = rand(Ns,1);
ph = 2*pi*rand(Ns,1);
rr = sqrt(max(0, 1-u.^2));
pts = [rr.*cos(ph), rr.*sin(ph), u];

scores = pts * seeds_xyz.';
[~, idx] = max(scores, [], 2);

k = sqrt(2./(1 + pts(:,3)));
xp = k .* pts(:,1);
yp = k .* pts(:,2);

figure('Color','w');
scatter(xp, yp, 6, idx, 'filled'); axis equal; grid on;
hold on;
k0 = sqrt(2./(1 + seeds_xyz(:,3)));
scatter(k0.*seeds_xyz(:,1), k0.*seeds_xyz(:,2), 80, 'k', 'filled');
title('Approx. spherical Voronoi cells on hemisphere (colored by nearest center)');
xlabel('x_{proj}'); ylabel('y_{proj}');

%% ==================== 2D PLOT: phi-cuts at theta_0 for each batch ========
% figure('Color','w'); hold on; grid on;
% 
% for b = 1:num_batches
%     % nearest theta index to this batch's theta0
%     [~, ith] = min(abs(theta_deg_range - theta0_list(b)));
%     G_cut = squeeze(G_base_all(ith,:,b));  % 1 x length(phi_deg_range)
% 
%     plot(phi_deg_range, G_cut);  % many curves on same plot
% 
%     % Optional: mark intended phi0 on this cut
%     % G_phi0 = interp1(phi_deg_range, G_cut, phi0_list(b), 'linear', 'extrap');
%     % plot(phi0_list(b), G_phi0, 'ko', 'MarkerSize', 3);
% end
% 
% xlabel('\phi (deg)');
% ylabel('Gain (dB)');
% title('Phi-cuts of base beams at corresponding \theta_0');

%% ==================== 2D PLOT: rect plot for each batch ========
% for b = 1:num_batches
%     G_b = G_base_all(:,:,b);
% 
%     % achieved peak
%     [~, idx] = max(G_b(:));
%     [ithp, iphp] = ind2sub(size(G_b), idx);
%     theta_p = theta_deg_range(ithp);
%     phi_p   = phi_deg_range(iphp);
% 
%     % intended (nearest grid point)
%     [~, ith0] = min(abs(theta_deg_range - theta0_list(b)));
%     [~, iph0] = min(abs(phi_deg_range   - phi0_list(b)));
%     theta_0g = theta_deg_range(ith0);
%     phi_0g   = phi_deg_range(iph0);
% 
%     figure('Color','w');
%     imagesc(phi_deg_range, theta_deg_range, G_b);
%     axis xy; grid on;
%     caxis([-40 0]);                    % IMPORTANT: clip dynamic range
%     colorbar;
%     xlabel('\phi (deg)'); ylabel('\theta (deg)');
% 
%     hold on;
%     plot(phi_0g, theta_0g, 'rx', 'MarkerSize', 12, 'LineWidth', 2); % target
%     plot(phi_p,  theta_p,  'wo', 'MarkerSize', 10, 'LineWidth', 2); % achieved peak
%     legend('','Target','Achieved peak','Location','best');
% 
%     title(sprintf('Batch %d: intended=[\\phi=%.2f, \\theta=%.2f], peak=[\\phi=%.2f, \\theta=%.2f]', ...
%         b, phi0_list(b), theta0_list(b), phi_p, theta_p));
% end

%% ==================== 3D PLOT: directions on the unit sphere =============
% Convert degrees to radians
% theta0_rad      = deg2rad(theta0_list);
% phi0_rad        = deg2rad(phi0_list);
% theta_peak_rad  = deg2rad(theta_peak_list);
% phi_peak_rad    = deg2rad(phi_peak_list);
% 
% % Target directions (unit vectors)
% x_t = sin(theta0_rad).*cos(phi0_rad);
% y_t = sin(theta0_rad).*sin(phi0_rad);
% z_t = cos(theta0_rad);
% 
% % Achieved peak directions
% x_p = sin(theta_peak_rad).*cos(phi_peak_rad);
% y_p = sin(theta_peak_rad).*sin(phi_peak_rad);
% z_p = cos(theta_peak_rad);
% 
% figure('Color','w'); hold on; grid on; axis equal;
% 
% % Draw a light sphere for reference
% [u,v] = meshgrid(linspace(0,2*pi,40), linspace(0,pi,20));
% xs = sin(v).*cos(u);
% ys = sin(v).*sin(u);
% zs = cos(v);
% surf(xs,ys,zs,'FaceAlpha',0.05,'EdgeColor',[0.8 0.8 0.8],'FaceColor',[0.9 0.9 0.9]);
% shading interp;
% 
% % Target directions (red)
% plot3(x_t, y_t, z_t, 'r.', 'MarkerSize', 18, 'DisplayName','Target');
% 
% % Achieved (blue)
% plot3(x_p, y_p, z_p, 'bo', 'MarkerSize', 6, 'DisplayName','Achieved peak');
% 
% xlabel('x'); ylabel('y'); zlabel('z');
% title('Target vs. achieved main-lobe directions on unit sphere');
% legend('Location','bestoutside');
% view(35,20);
% 
% %% ---------------------------- 3-D balloon (SURF) --------------------------
% for b = 1:num_batches
%     dynRange_dB = 40;
%     gain_dB_clipped = max(G_base_all(:,:,b), -dynRange_dB);
% 
%     % Radius mapping (choose one)
%     % Power radius (what you currently do):
%     R = 10.^(gain_dB_clipped/10);
% 
%     % Or amplitude radius (often looks better geometrically):
%     % R = 10.^(gain_dB_clipped/20);
% 
%     X = R .* sin(TH) .* cos(PH);
%     Y = R .* sin(TH) .* sin(PH);
%     Z = R .* cos(TH);
% 
%     figure('Color','w');
%     surf(X, Y, Z, gain_dB_clipped, 'EdgeColor','none');
%     axis equal vis3d; grid on;
%     xlabel('x'); ylabel('y'); zlabel('z');
%     title(sprintf('dx_{phase}=%.2f deg, dy_{phase}=%.2f deg, intended=[\\phi=%.2f, \\theta=%.2f]', ...
%         dx_base_all(b), dy_base_all(b), phi0_list(b), theta0_list(b)));
% 
%     cb = colorbar; ylabel(cb,'Gain (dB)');
%     lighting gouraud; camlight headlight;
% end


