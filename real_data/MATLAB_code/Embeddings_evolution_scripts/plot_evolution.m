function processAllSessions(full_trial_1, full_trial_2)
    % Process all sessions in land_off and no_land_off datasets
    processDataset(full_trial_2.sessions, 'full_trial_1');
    processDataset(full_trial_1.sessions, 'full_trial_2');
    disp('All session animations complete.');
end

function processDataset(sessions, datasetType)
    % Iterate through sessions in a dataset and process each session
    numSessions = length(sessions);
    
    for i = 1:numSessions
        session_idx = sessions{i}.session_idx; % Extract session index
        disp(['Processing ' datasetType ' session ' num2str(session_idx) '...']);
        processSession(sessions{i}, session_idx, datasetType);
    end
end

function processSession(sessionData, session_idx, datasetType)
    % Process a single session and save an animation as a .mp4 file

    close all;
    
    % Extract embeddings and velocity data
    embeddings_3d = squeeze(sessionData.embeddings_3d);
    binned_hipp_angle = sessionData.binned_hipp_angle;
    % binned_hipp_angle = binned_hipp_angle * (pi/180);
    binned_hipp_angle = mod(binned_hipp_angle, 2*pi);
    assignin('base','binned_hipp_angle',binned_hipp_angle);
    binned_vel = sessionData.binned_vel;
    binned_vel_rad = binned_vel * (2*pi/180);
    
    x = embeddings_3d(:, 1);
    y = embeddings_3d(:, 2);
    z = embeddings_3d(:, 3);
    nPoints = length(x);

    % Process binned hippocampal angle for colormap
    % binned_hipp_angle = mod(binned_hipp_angle, 2*pi); % Wrap around at 2π
    C = binned_hipp_angle; % Use as color data

    % Create figure
    fig = figure('KeyPressFcn', @keyPressCallback);
    hold on;
    grid on;
    title(['Session ' num2str(session_idx) ' Evolution']);
    xlabel('X Axis');
    ylabel('Y Axis');
    zlabel('Z Axis');
    xlim([min(x), max(x)]);
    ylim([min(y), max(y)]);
    zlim([min(z), max(z)]);
    view(3);

    % Initialize scatter plot
    scatterHandle = scatter3(x(1), y(1), z(1), 50, C(1), 'filled');

    % Colormap setup
    colormap('jet');
    cbar = colorbar;
    caxis([min(C), max(C)]);
    ylabel(cbar, 'Binned Hipp Angle (rad)', 'FontSize', 12, 'FontWeight', 'bold');

    % Store pause state in figure appdata
    setappdata(fig, 'isPaused', false);
    timeStep = 0.1; % 100 ms per frame

    % Add text annotation for velocity
    textPositionX = min(x) + 0.5;
    textPositionY = max(y) + 0.75;
    textPositionZ = max(z);
    valueText = text(textPositionX, textPositionY, textPositionZ, ...
        sprintf('Rat Velocity (rad/s): %.2f', binned_vel_rad(1)), ...
        'FontSize', 14, 'FontWeight', 'bold', 'Color', 'k');

    % Set correct frame rate for VideoWriter
    frameRate = 1 / timeStep; 
    videoFileName = sprintf('Velocity%sSession%dEvolution.mp4', datasetType, session_idx);
    videoFile = VideoWriter(videoFileName, 'MPEG-4');
    videoFile.FrameRate = frameRate;
    open(videoFile);

    % Animate points and save video
    for i = 1:nPoints
        while getappdata(fig, 'isPaused')
            pause(0.1);
        end
        
        scatter3(x(i), y(i), z(i), 50, C(i), 'filled');
        valueText.String = sprintf('Rat Velocity (rad/s): %.2f', binned_vel_rad(i));

        drawnow;

        % Save frame to video
        frame = getframe(gcf);
        writeVideo(videoFile, frame);
        
        pause(timeStep);
    end

    % Close video file
    close(videoFile);
    disp(['Animation saved: ' videoFileName]);
end

% Callback function for pausing and resuming animation
function keyPressCallback(~, event)
    if strcmp(event.Key, 'space') % Spacebar toggles pause
        isPaused = getappdata(gcf, 'isPaused');
        setappdata(gcf, 'isPaused', ~isPaused);
        if ~isPaused
            disp('Animation paused');
        else
            disp('Animation resumed');
        end
    end
end

processAllSessions(full_trial_1, full_trial_2);

