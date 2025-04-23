function processAllSessions(full_trial_1, full_trial_2, allowedSessions, keepTrail)
    processDataset(full_trial_1.sessions, 'full_trial_2', allowedSessions, keepTrail);
    % processDataset(full_trial_2.sessions, 'full_trial_2', allowedSessions, keepTrail);
    disp('All requested session animations complete.');
end

function processDataset(sessions, datasetType, allowedSessions, keepTrail)
    for k = 1:numel(allowedSessions)
        sid = allowedSessions(k);
        idx = find(cellfun(@(s) s.session_idx == sid, sessions), 1, 'first');
        if isempty(idx)
            warning('Session %d not found in %s – skipping.', sid, datasetType);
            continue
        end
        fprintf('Processing %s session %d …\n', datasetType, sid);
        processSession(sessions{idx}, sid, datasetType, keepTrail);
    end
end

% -------------------------------------------------------------------------
function processSession(sessionData, session_idx, datasetType, keepTrail)
    close all;

    % ---------------------------------------------------------------------
    % Load & merge embeddings
    embeddings_3d      = squeeze(sessionData.embeddings_3d);       % N×3
    embeddings_low_vel = squeeze(sessionData.embeddings_low_vel);  % (N–M)×3
    high_vel_mask      = sessionData.high_vel_mask;                % N×1 logical

    assert(sum(~high_vel_mask)==size(embeddings_low_vel,1), ...
        'Low‐vel mask count must match rows in embeddings_low_vel');

    combined_embeddings = embeddings_3d;
    combined_embeddings(~high_vel_mask, :) = embeddings_low_vel;

    x = combined_embeddings(:,1);
    y = combined_embeddings(:,2);
    z = combined_embeddings(:,3);

    binned_hipp_angle = mod(sessionData.binned_hipp_angle, 2*pi);
    binned_hipp_angle(isnan(binned_hipp_angle)) = 0;
    binned_vel         = sessionData.binned_vel;
    C = binned_hipp_angle;
    nPts = numel(x);

    lowVelRGB = [1 0.45 0.75];

    % ---------------------------------------------------------------------
    % Figure
    fig = figure('Visible','on', 'Units','pixels', 'Position',[100 100 1120 840]);
    rotate3d(fig,'on');
    hold on; grid on; view(3);
    xlabel('X'); ylabel('Y'); zlabel('Z');
    title(sprintf('Session %d Evolution', session_idx));
    xlim([min(x) max(x)]);
    ylim([min(y) max(y)]);
    zlim([min(z) max(z)]);

    colormap('jet');
    cbar = colorbar;
    caxis([min(C) max(C)]);
    ylabel(cbar,'Binned Hipp Angle (rad)','FontSize',12,'FontWeight','bold');

    % plot principal curve
    pc = squeeze(sessionData.principal_curves_3d);  % 201×3
    plot3(pc(:,1), pc(:,2), pc(:,3), 'r-', 'LineWidth', 2);

    % display velocity and timestamp as static annotations
    valueText = annotation('textbox', [0.02 0.90 0.3 0.05], ...
        'String', sprintf('Rat Velocity (rad/s): %.2f', binned_vel(1)), ...
        'FontSize',14, 'FontWeight','bold', 'EdgeColor','none', 'FitBoxToText','on');
    timeText  = annotation('textbox', [0.02 0.84 0.3 0.05], ...
        'String', 'Time: 0 s', ...
        'FontSize',14, 'FontWeight','bold', 'EdgeColor','none', 'FitBoxToText','on');

    % ---------------------------------------------------------------------
    % Scatter handles
    if keepTrail
        scatHigh = scatter3(NaN,NaN,NaN,50,'filled','MarkerEdgeColor','none');
        scatLow  = scatter3(NaN,NaN,NaN,50,lowVelRGB,'filled','MarkerEdgeColor','none');
        hvx = []; hvy = []; hvz = []; hvC = [];
        lvx = []; lvy = []; lvz = [];
    else
        scat = scatter3(NaN,NaN,NaN,50,'filled','MarkerEdgeColor','none');
    end

    % ---------------------------------------------------------------------
    % Video writer
    dt  = 0.1;  % seconds/frame
    vw = VideoWriter(sprintf('Velocity%sSession%dEvolution.mp4', datasetType, session_idx), 'MPEG-4');
    vw.FrameRate = 1/dt;
    open(vw);

    % ---------------------------------------------------------------------
    % Animation loop
    for idx = 1:nPts
        if keepTrail
            if high_vel_mask(idx)
                hvx(end+1)=x(idx); hvy(end+1)=y(idx); hvz(end+1)=z(idx); hvC(end+1)=C(idx);
                set(scatHigh, 'XData',hvx,'YData',hvy,'ZData',hvz,'CData',hvC(:),'MarkerFaceColor','flat');
            else
                lvx(end+1)=x(idx); lvy(end+1)=y(idx); lvz(end+1)=z(idx);
                set(scatLow, 'XData',lvx,'YData',lvy,'ZData',lvz,'CData',repmat(lowVelRGB, numel(lvx),1));
            end
        else
            if high_vel_mask(idx)
                set(scat, 'XData',x(idx),'YData',y(idx),'ZData',z(idx),'CData',C(idx),'MarkerFaceColor','flat');
            else
                set(scat, 'XData',x(idx),'YData',y(idx),'ZData',z(idx),'CData',lowVelRGB,'MarkerFaceColor',lowVelRGB);
            end
        end

        % update static annotations
        valueText.String = sprintf('Rat Velocity: %.2f', binned_vel(idx));
        timeText.String  = sprintf('Time: %d s', idx);

        drawnow limitrate nocallbacks;
        writeVideo(vw, getframe(fig));
        pause(0.5)
    end

    close(vw);
    close(fig);
    fprintf('Saved: Velocity%sSession%dEvolution.mp4\n', datasetType, session_idx);
end

% -------------------------------------------------------------------------
% Example usage:
allowed   = [36];
keepTrail = true;
processAllSessions(full_trial_2, full_trial_2, allowed, keepTrail);
