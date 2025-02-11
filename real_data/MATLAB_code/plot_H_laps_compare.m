function lap_data = plotHs(decode_H, binned_est_gain, lap_number, session_idx, dataset_name, overlay_laps, aggregate_mode, bin_size)
% plotHs plots decoded trial data in one of three modes:
%
%   Standard Mode (overlay_laps == false and aggregate_mode == false):
%     Plots decode_H and binned_est_gain vs. lap_number.
%
%   Overlay Mode (overlay_laps == true and aggregate_mode == false):
%     Separates each lap, normalizes the lap time to [0,1], and overlays the laps.
%
%   Aggregated Mode (aggregate_mode == true):
%     Bins the normalized lap time (using a variable bin size, default = 0.05)
%     and computes the mean and standard deviation of decode_H across laps.
%     Only decode_H is used in this mode.
%
% Inputs:
%   decode_H       - Numerical array of decoded H values.
%   binned_est_gain- Numerical array of estimated values (used only in Standard and Overlay modes).
%   lap_number     - Numerical array (float) representing lap progression.
%   session_idx    - Integer representing the session number.
%   dataset_name   - String representing the dataset name.
%   overlay_laps   - Boolean flag; if true, overlay all laps (aligned to 0-1).
%   aggregate_mode - Boolean flag; if true, produce aggregated binned plot of decode_H.
%   bin_size       - (Optional) Scalar specifying the bin width (default = 0.05).
%
% The figure is saved in the folder "decoded_H_vs_binned_gain_plots".

    % Set default values for optional arguments
    if nargin < 7
        aggregate_mode = false;
    end
    if nargin < 8
        bin_size = 0.05;
    end

    % Create output directory if it does not exist.
    out_folder = 'decoded_H_vs_binned_gain_plots';
    if ~exist(out_folder, 'dir')
        mkdir(out_folder);
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Aggregated Mode: Bin normalized lap times and aggregate decode_H.
    if aggregate_mode
        fig = figure('Visible', 'on');
        hold on;
        
        % Define bin edges and centers (variable bin size)
        bin_edges = 0:bin_size:1;
        bin_centers = bin_edges(1:end-1) + bin_size/2;
        num_bins = length(bin_edges) - 1;
        
        % Identify unique laps using the integer part of lap_number
        unique_laps = unique(floor(lap_number));
        num_laps = length(unique_laps);
        
        % Preallocate matrix to store binned means for each lap
        lap_bin_means = NaN(num_laps, num_bins);
        
        % Loop over each lap and compute the mean decode_H in each bin.
        for j = 1:num_laps
            lap_val = unique_laps(j);
            % Get indices for this lap
            idx = (lap_number >= lap_val) & (lap_number < lap_val + 1);
            % Normalize lap time to [0,1]
            norm_time = lap_number(idx) - lap_val;
            lap_decode_H = decode_H(idx);

            % Normalize each lap's decode_H by subtracting its mean.
            lap_mean = mean(lap_decode_H); % Compute the mean of this lap's decode_H
            lap_decode_H = lap_decode_H - lap_mean; % Normalize by mean
            
            % Bin the data within this lap.
            for i = 1:num_bins
                % Find indices within the current bin.
                bin_idx = (norm_time >= bin_edges(i)) & (norm_time < bin_edges(i+1));
                if any(bin_idx)
                    lap_bin_means(j, i) = mean(lap_decode_H(bin_idx));
                end
            end
        end
        
        % Compute aggregated mean and standard deviation across laps for each bin.
        agg_mean = nanmean(lap_bin_means, 1);   % mean across laps (ignoring NaNs)
        agg_std  = nanstd(lap_bin_means, 0, 1);   % standard deviation across laps
        
        %%% NEW CODE: Determine the bin with the minimum standard deviation
        [min_std, min_idx] = min(agg_std);
        min_bin = bin_centers(min_idx);
        min_mean = agg_mean(min_idx);
        %%% End of new code
        
        % Plot the aggregated mean with error bars (std).
        errorbar(bin_centers, agg_mean, agg_std, 'o-', 'LineWidth', 2);
        
        xlabel('Normalized Lap Time (0 to 1)');
        ylabel('Decoded H');
        title(sprintf('%s: Aggregated Decoded H vs Normalized Lap Time (Session %d)', dataset_name, session_idx));
        hold off;
        
        % Save the figure.
        save_path = fullfile(out_folder, sprintf('%s_session_%d_aggregated.png', dataset_name, session_idx));
        saveas(fig, save_path);

        % Set output data
        lap_data.bin_centers  = bin_centers;
        lap_data.agg_mean     = agg_mean;
        lap_data.agg_std      = agg_std;
        lap_data.lap_bin_means = lap_bin_means;
        % Return the bin with the minimum std, its std, and its mean.
        lap_data.min_bin      = min_bin;
        lap_data.min_std      = min_std;
        lap_data.min_mean     = min_mean;
        
        return;
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Standard and Overlay Modes (keeping old functionality)
    if ~overlay_laps
        %% Standard Plot Mode: Plot vs. lap_number (both decode_H and binned_est_gain)
        fig = figure('Visible', 'off');
        hold on;
        plot(lap_number, decode_H, 'b-', 'LineWidth', 2);
        plot(lap_number, binned_est_gain, 'r-', 'LineWidth', 2);
        hold off;
        
        xlabel('Lap Number (Fractional)');
        ylabel('Value');
        title(sprintf('%s: Decoded H vs Fourier H (Session %d)', dataset_name, session_idx));
        legend('Decoded H', 'Binned Est Gain', 'Location', 'best');
        
        save_path = fullfile(out_folder, sprintf('%s_session_%d.png', dataset_name, session_idx));
        saveas(fig, save_path);
        
        lap_data = [];
        
    else
        %% Overlay Plot Mode: Separate and align each lap (normalized time 0-1)
        fig = figure('Visible', 'on');
        hold on;
        
        % Identify full lap indices
        min_lap = floor(min(lap_number));
        max_lap = floor(max(lap_number));
        
        % Store lap data in a cell array
        lap_data = cell(1, max_lap - min_lap + 1);
        
        for lap = min_lap:max_lap
            % Extract data for the current lap.
            idx = (lap_number >= lap) & (lap_number < lap + 1);
            % Skip laps with too few data points.
            if sum(idx) < 2
                continue;
            end
            
            % Normalize lap time to [0,1]
            norm_lap_time = lap_number(idx) - lap;
            
            % Store data as [normalized_time; decode_H; binned_est_gain]
            lap_data{lap - min_lap + 1} = [norm_lap_time; decode_H(idx); binned_est_gain(idx)];
        end
        
        % Plot all laps with distinct colors.
        colors = lines(length(lap_data));
        for i = 1:length(lap_data)
            if isempty(lap_data{i})
                continue;
            end
            lap_array = lap_data{i};
            x_vals = lap_array(1, :);
            decode_H_vals = lap_array(2, :);
            % Optionally smooth the trace.
            decode_H_vals = smooth(decode_H_vals - mean(decode_H_vals), 5);
            
            % Plot decode_H as a solid line.
            plot(x_vals, decode_H_vals, '-', 'Color', colors(i, :), 'LineWidth', 1.5);
        end
        hold off;
        
        xlabel('Normalized Lap Time (0 to 1)');
        ylabel('Value');
        title(sprintf('%s: Aligned Lap Overlay (Session %d)', dataset_name, session_idx));
        legend({'Decoded H (solid)', 'Binned Est Gain (dashed)'}, 'Location', 'best');
        
        save_path = fullfile(out_folder, sprintf('%s_session_%d_overlay.png', dataset_name, session_idx));
        saveas(fig, save_path);
    end
end

% Example usage:
for i = 8:12
    lap_data = plotHs(full_trial_2.sessions{i}.lap_decode_H, ...
                      full_trial_2.sessions{i}.lap_est_gain, ...
                      full_trial_2.sessions{i}.lap_number, ...
                      full_trial_2.sessions{i}.session_idx, ...
                      'Full Trial True Laps', true, true, 0.05);
    % Now lap_data contains:
    %   - lap_data.min_bin: the bin center with the minimum standard deviation
    %   - lap_data.min_std: the minimum standard deviation
    %   - lap_data.min_mean: the aggregated mean for that bin
end
