function lap_data = plotHs(decode_H, binned_est_gain, lap_number, session_idx, dataset_name, overlay_laps)
% plotHs plots decoded trial data in one of two modes:
%
%   Standard Mode (overlay_laps = false):
%     Plots decode_H and binned_est_gain vs. lap_number.
%
%   Overlay Mode (overlay_laps = true):
%     Extracts each lap separately while keeping fractional positions
%     and aligns them to a common x-axis (0-1) for visualization.
%
% Inputs:
%   decode_H       - Numerical array of decoded values.
%   binned_est_gain- Numerical array of estimated values.
%   lap_number     - Numerical array (float) representing lap progression.
%   session_idx    - Integer representing the session number.
%   dataset_name   - String representing the dataset name.
%   overlay_laps   - Boolean flag; if true, overlay all laps (aligned to 0-1).
%
% The figure is saved in the folder "decoded_H_vs_binned_gain_plots".
%

    % Create output directory if it does not exist.
    out_folder = 'decoded_H_vs_binned_gain_plots';
    if ~exist(out_folder, 'dir')
        mkdir(out_folder);
    end

    if ~overlay_laps
        %% Standard Plot Mode: Plot vs. lap_number
        fig = figure('Visible', 'off');
        hold on;
        plot(lap_number, decode_H, 'b-', 'LineWidth', 2);
        plot(lap_number, binned_est_gain, 'r-', 'LineWidth', 2);
        hold off;
        
        xlabel('Lap Number (Fractional)');
        ylabel('Value');
        title(sprintf('%s: Manifold Decoded H vs Fourier H (Session %d)', dataset_name, session_idx));
        legend('Decode H', 'Binned Est Gain', 'Location', 'best');
        
        save_path = fullfile(out_folder, sprintf('%s_session_%d.png', dataset_name, session_idx));
        saveas(fig, save_path);
        close(fig);
    else
        %% Overlay Plot Mode: Separate and align each lap
        fig = figure('Visible', 'on');
        hold on;
        
        % Identify full lap indices
        min_lap = floor(min(lap_number));
        max_lap = floor(max(lap_number));
        
        % Store lap data
        lap_data = cell(1, max_lap - min_lap + 1);

        lap_data

        min_lap
        max_lap
        
        for lap = min_lap:max_lap
            % Extract data where lap_number is within [lap, lap+1)
            idx = (lap_number >= lap) & (lap_number < lap + 1);
            % Skip laps with too few data points
            if sum(idx) < 2
                continue;
            end
            
            % Normalize lap_number within [0,1]
            norm_lap_time = lap_number(idx) - lap; % Keeps fractional progress
            
            % Store data as [normalized_time; decode_H; binned_est_gain]
            lap_data{lap - min_lap + 1} = [norm_lap_time; decode_H(idx); binned_est_gain(idx)];
        end

        lap_data

        % Plot all laps with distinct colors
        colors = lines(length(lap_data));
        for i = 20:5:40
            if isempty(lap_data{i})
                continue;
            end
            lap_array = lap_data{i};
            x_vals = lap_array(1, :);
            disp(length(x_vals))
            decode_H_vals = lap_array(2, :);
            decode_H_vals = smooth(decode_H_vals - mean(decode_H_vals),5);
            % binned_est_gain_vals = lap_array(3, :);

            % Plot decode_H as a solid line
            plot(x_vals, decode_H_vals, '-', 'Color', colors(i, :), 'LineWidth', 1.5);
            % Plot binned_est_gain as a dashed line
            % plot(x_vals, binned_est_gain_vals, '--', 'Color', colors(i, :), 'LineWidth', 1.5);
        end
        hold off;
        
        xlabel('Normalized Lap Time (0 to 1)');
        ylabel('Value');
        title(sprintf('%s: Aligned Lap Overlay (Session %d)', dataset_name, session_idx));
        legend({'Decode H (solid)', 'Binned Est Gain (dashed)'}, 'Location', 'best');
        
        save_path = fullfile(out_folder, sprintf('%s_session_%d_overlay.png', dataset_name, session_idx));
        saveas(fig, save_path);
    end
end


% Standard plot over lap numbers
% plotHs(full_trial_1.sessions{8}.lap_decode_H, ...
%        full_trial_1.sessions{8}.lap_est_gain, ...
%        full_trial_1.sessions{8}.lap_number, ...
%        full_trial_1.sessions{8}.session_idx, ...
%        'Full Trial', false);

% Overlay plot with offset x-axis
lap_data = plotHs(full_trial_1.sessions{8}.lap_decode_H, ...
       full_trial_1.sessions{8}.lap_est_gain, ...
       full_trial_1.sessions{8}.lap_number, ...
       full_trial_1.sessions{8}.session_idx, ...
       'Full Trial', true);
