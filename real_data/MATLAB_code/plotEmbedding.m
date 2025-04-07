function processSessionStatic(sessionData, session_idx, datasetType)
    % Plot 3D embeddings as a static scatter plot with color from hippocampal angle

    % Extract embeddings and hippocampal angle
    embeddings_3d = squeeze(sessionData.embeddings_3d);
    binned_hipp_angle = sessionData.binned_vel;  
    binned_hipp_angle = mod(binned_hipp_angle, 2*pi); % wrap to [0, 2π]

    x = embeddings_3d(:, 1);
    y = embeddings_3d(:, 2);
    z = embeddings_3d(:, 3);
    C = binned_hipp_angle;

    % Create figure
    figure;
    scatter3(x, y, z, 50, C, 'filled');
    title(sprintf('3D Embeddings — Session %d', session_idx));
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    grid on;
    axis tight;
    colormap('jet');
    colorbar;
    caxis([0, 2*pi]);
    view(3);  % 3D view
end

function processDataset(sessions, datasetType)
    numSessions = length(sessions);
    
    for i = 1:numSessions
        session_idx = sessions{i}.session_idx;
        disp(['Plotting ' datasetType ' session ' num2str(session_idx) '...']);
        processSessionStatic(sessions{i}, session_idx, datasetType);
    end
end

processDataset(full_trial_2.sessions, 'full_trial_1');