close all;
% Extract embeddings
idxs = find(full_trial_1(1).sessions{1}.SI_score_hipp > 0.75)
session_idxs = session_idx(idxs)
for i = 1:length(idxs)
    idx = idxs(i)
    session_idx = session_idxs(i);
    embeddings = embeddings_3d{idx};
    hipp_angle = binned_hipp_angle{idx};
    high_vel = binned_high_vel{idx};
    %binned_high_vel_rad = binned_high_vel * (2*pi/180);
    x = embeddings(:, 1);
    y = embeddings(:, 2);
    z = embeddings(:, 3);
    nPoints = length(x);
    
    % Process binned hippocampal angle for colormap
    hipp_angle = mod(hipp_angle, 2*pi); % Mod 2*pi to wrap around
    C = hipp_angle; % Use as color data
    
    % Create the figure for the animation
    fig = figure('KeyPressFcn', @keyPressCallback); % Attach callback for pausing
    hold on;
    grid on;
    title('Session Evolution Over Time');
    xlabel('X Axis');
    ylabel('Y Axis');
    zlabel('Z Axis');
    xlim([min(x), max(x)]); % Adjust according to your data range
    ylim([min(y), max(y)]); % Adjust according to your data range
    zlim([min(z), max(z)]); % Adjust according to your data range
    view(3);
    
    % Initialize the scatter plot
    scatterHandle = scatter3(x(1), y(1), z(1), 50, C(1), 'filled');
    
    % Colormap and colorbar setup
    colormap('jet'); % Apply the colormap
    cbar = colorbar; % Add a colorbar to show the color scale
    caxis([min(C), max(C)]); % Match color scale to the range of C
    ylabel(cbar, 'Binned Hipp Angle (rad)', 'FontSize', 12, 'FontWeight', 'bold'); % Label the colorbar
    
    % Store pause state in the figure's appdata
    setappdata(fig, 'isPaused', false);
    
    % Time step (100ms = 0.1 seconds)
    timeStep = 0.01;
    
    % Add text annotation for displaying high_vel values
    textPositionX = min(x) + 0.5; % Offset from data points
    textPositionY = max(y) + 0.75;
    textPositionZ = max(z);
    valueText = text(textPositionX, textPositionY, textPositionZ, ...
        sprintf('Rat Velocity (deg/s): %.2f', high_vel(1)), ...
        'FontSize', 14, 'FontWeight', 'bold', 'Color', 'k');
    
    % Set the correct frame rate for video (1/timeStep)
    frameRate = 1 / timeStep; 
    videoFileName = sprintf('SessionEvolution_Session%d.mp4', session_idx);
    videoFile = VideoWriter(videoFileName, 'MPEG-4');
    videoFile.FrameRate = frameRate; % Match animation speed
    open(videoFile);
    
    % Animate points and save video
    for i = 1:nPoints
        % Check if the animation is paused
        while getappdata(fig, 'isPaused')
            pause(0.1); % Wait until unpaused
        end
    
        % Add the next point to the scatter plot
        scatter3(x(i), y(i), z(i), 50, C(i), 'filled');
        
        % Update the text annotation with the current high_vel value
        valueText.String = sprintf('Rat Velocity (deg/s): %.2f', high_vel(i));
    
        % Update the plot dynamically
        drawnow;
    
        % Save the frame to video
        frame = getframe(gcf);
        writeVideo(videoFile, frame);
    
        % Pause for the time step (only for live animation, not for video)
        pause(timeStep);
    end
    
    % Close the video file after saving
    close(videoFile);
    disp('Animation complete and saved as SessionEvolutionWithColormap.mp4');
end

% Callback function for pausing and resuming animation
function keyPressCallback(~, event)
    % Access the figure's appdata to toggle the pause state
    if strcmp(event.Key, 'space') % Check for spacebar press
        isPaused = getappdata(gcf, 'isPaused'); % Get current state
        setappdata(gcf, 'isPaused', ~isPaused); % Toggle state
        if ~isPaused
            disp('Animation paused');
        else
            disp('Animation resumed');
        end
    end
end
