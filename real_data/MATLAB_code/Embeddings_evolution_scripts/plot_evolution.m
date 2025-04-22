function processAllSessions(full_trial_1, full_trial_2, allowedSessions)
    processDataset(full_trial_2.sessions, 'full_trial_1', allowedSessions);
    processDataset(full_trial_1.sessions, 'full_trial_2', allowedSessions);
    disp('All requested session animations complete.');
end

function processDataset(sessions, datasetType, allowedSessions)
    % Iterate through sessions in a dataset and process each session
    for i = 1:length(sessions)
        sid = sessions{i}.session_idx;
        if ~ismember(sid, allowedSessions)
            continue  % skip this session
        end
        fprintf('Processing %s session %d...\n', datasetType, sid);
        processSession(sessions{i}, sid, datasetType);
    end
end

function processSession(sessionData, session_idx, datasetType)
    % (your existing code, unchanged)
    close all;
    
    embeddings_3d     = squeeze(sessionData.embeddings_3d);
    binned_hipp_angle = mod(sessionData.binned_hipp_angle, 2*pi);
    binned_vel_rad    = sessionData.binned_vel * (2*pi/180);
    high_vel_mask     = sessionData.high_vel_mask;
    
    x = embeddings_3d(:,1);
    y = embeddings_3d(:,2);
    z = embeddings_3d(:,3);
    nPoints = numel(x);
    C = binned_hipp_angle;

    %% Figure setup (visible)
    fig = figure('Visible','on');
    hold on; grid on;
    title(sprintf('Session %d Evolution', session_idx));
    xlabel('X'); ylabel('Y'); zlabel('Z');
    xlim([min(x) max(x)]); ylim([min(y) max(y)]); zlim([min(z) max(z)]);
    view(3);

    colormap('jet');
    cbar = colorbar;
    caxis([min(C) max(C)]);
    ylabel(cbar,'Binned Hipp Angle (rad)','FontSize',12,'FontWeight','bold');

    valueText = text(min(x)+0.5, max(y)+0.75, max(z), ...
        sprintf('Rat Velocity (rad/s): %.2f', binned_vel_rad(1)), ...
        'FontSize',14,'FontWeight','bold','Color','k');

    %% Video writer
    timeStep = 0.1;               % 100 ms/frame
    videoFile = VideoWriter( ...
      sprintf('Velocity%sSession%dEvolution.mp4', datasetType, session_idx), ...
      'MPEG-4');
    videoFile.FrameRate = 1/timeStep;
    open(videoFile);

    %% Animate & save
    for idx = 1:nPoints
        if high_vel_mask(idx)
            scatter3(x(idx), y(idx), z(idx), 50, C(idx), 'filled');
        else
            scatter3(x(idx), y(idx), z(idx), 50, [1 0.75 0.8], 'filled');
        end
        valueText.String = sprintf('Rat Velocity (rad/s): %.2f', binned_vel_rad(idx));
        drawnow;
        writeVideo(videoFile, getframe(gcf));
        pause(timeStep);
    end

    close(videoFile);
    close(fig);
    fprintf('Saved: Velocity%sSession%dEvolution.mp4\n', datasetType, session_idx);
end


allowed = [38 35];
processAllSessions(full_trial_2, full_trial_2, allowed);
