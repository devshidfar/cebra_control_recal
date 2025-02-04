function plotTwo3DArrays(array1, array2, save_path)
    % Ensure arrays have the correct dimensions
    if size(array1, 2) ~= 3 || size(array2, 2) ~= 3
        error('Both arrays must have 3 columns for (X, Y, Z) coordinates.');
    end

    % Extract X, Y, Z coordinates for each array
    x1 = array1(:, 1); y1 = array1(:, 2); z1 = array1(:, 3);
    x2 = array2(:, 1); y2 = array2(:, 2); z2 = array2(:, 3);

    % Create figure
    fig = figure('Visible', 'off'); % Create figure but don't display it
    hold on;
    grid on;
    
    % Plot first array in red
    scatter3(x1, y1, z1, 50, 'r', 'filled');

    % Plot second array in blue
    scatter3(x2, y2, z2, 50, 'b', 'filled');

    % Labels and title
    xlabel('X Axis');
    ylabel('Y Axis');
    zlabel('Z Axis');
    title('3D Scatter Plot of Two Arrays');

    % Legend
    legend('Landmarks On (Red)', 'Landmarks Off (Blue)', 'Location', 'best');

    % Adjust view
    view(3);
    axis equal;
    
    hold off;

    % Save the figure
    saveas(fig, save_path);
    
    % Close figure to free memory
    close(fig);
end

% Create the folder to store plots
output_folder = 'land_on_land_off_plots';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% Loop through sessions
for i = 1:25
    session_idx = land_on.sessions{i}.session_idx; % Get session index

    % Get number of rows in embeddings_3d
    land_on_embeddings_idx = size(land_on.sessions{i}.embeddings_3d, 1);

    % Extract land_on and land_off embeddings
    land_on_embeddings = full_trial.sessions{i}.embeddings_3d(1:land_on_embeddings_idx, :);
    land_off_embeddings = full_trial.sessions{i}.embeddings_3d(land_on_embeddings_idx + 1:end, :);

    % Construct the file path to save the plot
    save_path = fullfile(output_folder, sprintf('session_%d.png', session_idx));

    % Plot and save the figure
    plotTwo3DArrays(land_on_embeddings, land_off_embeddings, save_path);
end
