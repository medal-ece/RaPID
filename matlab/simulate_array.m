function G_dB = simulate_array(M,N,dx_phase_deg,dy_phase_deg, ...
                               elem_gain_lin,TH,PH,dx,dy,k,normalize_to_0dB)
    % Progressive phase across a centered MxN array, driven by dx_phase, dy_phase (deg)
    Beta_deg = zeros(M,N);
    for m = 1:M
        for n = 1:N
            mx = m - (M+1)/2;    % index offset in x
            ny = n - (N+1)/2;    % index offset in y
            Beta_deg(m,n) = mx*dx_phase_deg + ny*dy_phase_deg;
        end
    end
    A_w = ones(M,N);             % equal amplitudes
    G_dB = local_array_pattern(M,N,Beta_deg,A_w, ...
        elem_gain_lin,TH,PH,dx,dy,k,normalize_to_0dB);
end

%% ==================== Local array pattern function (unchanged) ===========
function G_dB = local_array_pattern(M,N,Beta_deg,A_w, ...
    elem_gain_lin,TH,PH,dx,dy,k,normalize_to_0dB)

    % Generalized (centered) AF with arbitrary per-element phase Beta_deg and amplitude A_w
    Beta = deg2rad(Beta_deg);                      % M×N
    if isscalar(A_w), A_w = A_w*ones(M,N); end
    xm = ((0:M-1) - (M-1)/2) * dx;                 % element x positions
    yn = ((0:N-1) - (N-1)/2) * dy;                 % element y positions
    AF = zeros(size(TH));
    for m = 1:M
        for n = 1:N
            phase = k*( xm(m).*sin(TH).*cos(PH) + yn(n).*sin(TH).*sin(PH) ) + Beta(m,n);
            AF = AF + A_w(m,n).*exp(1j*phase);
        end
    end
    AF_pow = abs(AF).^2;
    P = elem_gain_lin .* AF_pow;                   % power pattern
    G_dB = 10*log10(P + eps);
    if normalize_to_0dB, G_dB = G_dB - max(G_dB(:)); end
end