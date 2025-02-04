% Store datasets in a struct to avoid conflicts with 'full'
dataset_structs.land_on = land_on;
dataset_structs.full_trial = full_trial; % Rename to avoid conflict with MATLAB's 'full'
dataset_structs.train_land_on = train_land_on;

% Define dataset names
datasets = {'full_trial'}; 

% Define output folder
output_folder = fullfile(pwd, 'continuous_traces'); % Save in a folder in the same directory

% Create folder if it doesn't exist
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% Toggle between heatmap and line plot mode
use_heatmap = false; % Set to true for heatmap, false for line plot

% Loop over sessions
for i = 1:length(land_on.sessions)
    session_idx = land_on.sessions{i}.session_idx; % Extract session index

    % Loop over datasets
    for d = 1:length(datasets)
        dataset_name = datasets{d}; % Get dataset name

        % Access dataset using struct (avoid eval)
        dataset_struct = dataset_structs.(dataset_name);
        neural_data = dataset_struct.sessions{i}.neural_data_full_trial; % Access neural_data

        % Check if neural_data exists and is not empty
        if isempty(neural_data)
            fprintf('[WARNING] Skipping session %d for %s: neural_data is empty.\n', session_idx, dataset_name);
            continue;
        end

        % Define filename for saving in the folder
        filename = fullfile(output_folder, sprintf('neural_plot_%s_session_%d.png', dataset_name, session_idx));

        % Create figure (but do NOT display it)
        fig = figure('Visible', 'off'); % Prevent figure pop-ups
        
        if use_heatmap
            % HEATMAP MODE
            imagesc(neural_data'); % Transpose: time (x-axis), neurons (y-axis)
            colormap('hot'); % Set colormap
            colorbar; % Add colorbar
            xlabel('Time');
            ylabel('Neuron Index');
            title(sprintf('Neural Data Heatmap - %s - Session %d', dataset_name, session_idx));
        else
            % LINE PLOT MODE
            hold on;
            T = size(neural_data, 1); % Get time dimension
            N = size(neural_data, 2); % Get number of neurons
            
            % Normalize data for better visibility
            offset = 10; % Space between traces
            for n = 1:N
                plot(1:T, neural_data(:, n) + offset * (n - 1), 'k'); % Stack traces
            end

            xlabel('Time');
            ylabel('Neuron Index');
            title(sprintf('Neural Traces - %s - Session %d', dataset_name, session_idx));
            hold off;
        end

        % Save the figure
        saveas(fig, filename);
        close(fig); % Close figure to free memory
    end
end
