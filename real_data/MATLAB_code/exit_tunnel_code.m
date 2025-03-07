%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Overlay Plot Mode: Separate and align each lap (normalized time 0-1)
if overlay_laps && ~aggregate_mode
    fig = figure('Visible', 'on');
    hold on;
    
    % Identify full lap indices
    min_lap = floor(min(lap_number));
    max_lap = floor(max(lap_number));
    
    % Store lap data in a cell array.
    % Each nonempty cell will contain a 3-by-N matrix:
    % [normalized_lap_time; decode_H; binned_est_gain]
    lap_data = cell(1, max_lap - min_lap + 1);
    valid_lap_idx = []; % to keep track of lap numbers (for labeling)
    for lap = min_lap:max_lap
        % Extract data for the current lap.
        idx = (lap_number >= lap) & (lap_number < lap + 1);
        % Skip laps with too few data points.
        if sum(idx) < 2
            continue;
        end
        
        % Normalize lap time to [0,1]
        norm_lap_time = lap_number(idx) - lap;
        lap_decode_H = decode_H(idx);
        lap_est_gain = binned_est_gain(idx);
        
        % (Optional) Normalize each lap’s decode_H by subtracting its mean.
        lap_decode_H = lap_decode_H - mean(lap_decode_H);
        
        % Store data as a row vector for each quantity.
        lap_data{lap - min_lap + 1} = [norm_lap_time'; lap_decode_H'; lap_est_gain'];
        valid_lap_idx = [valid_lap_idx, lap - min_lap + 1];
    end
    % Remove empty cells (if any)
    lap_data = lap_data(~cellfun('isempty', lap_data));
    
    % Plot all laps with distinct colors.
    colors = lines(length(lap_data));
    for i = 1:length(lap_data)
        lap_array = lap_data{i};
        x_vals = lap_array(1, :);
        decode_H_vals = lap_array(2, :);
        % Optionally smooth the trace.
        decode_H_vals = smooth(decode_H_vals, 5);
        plot(x_vals, decode_H_vals, '-', 'Color', colors(i, :), 'LineWidth', 1.5);
    end
    hold off;
    
    xlabel('Normalized Lap Time (0 to 1)');
    ylabel('Decoded H');
    title(sprintf('%s: Aligned Lap Overlay (Session %d)', dataset_name, session_idx));
    legend(arrayfun(@(x) sprintf('Lap %d', x), valid_lap_idx, 'UniformOutput', false), 'Location', 'best');
    
    save_path = fullfile(out_folder, sprintf('%s_session_%d_overlay.png', dataset_name, session_idx));
    saveas(fig, save_path);
    
    % ---------------- Alignment Distance Analysis ----------------
    % We assume the bin size (e.g., 0.05) defines a fixed grid over [0,1]
    % onto which we interpolate each lap’s decoded H values.
    x_grid = 0:bin_size:1;
    num_bins = length(x_grid);
    num_laps = length(lap_data);
    
    % Preallocate a matrix to store the interpolated H traces.
    H_matrix = NaN(num_laps, num_bins);
    for i = 1:num_laps
        lap_array = lap_data{i};
        norm_time = lap_array(1, :);
        decode_H_vals = lap_array(2, :);
        % Interpolate the decoded H values onto x_grid (using linear interpolation).
        H_matrix(i, :) = interp1(norm_time, decode_H_vals, x_grid, 'linear', 'extrap');
    end
    
    % Set parameters for the alignment metric (using bin indices):
    % l_max: maximum allowed shift (in bins)
    % epsilon: penalty coefficient for the shift
    % p: exponent for the L^p norm (e.g., 2 for Euclidean)
    l_max = 2;      % adjust as needed
    epsilon = 0.1;  % adjust as needed
    p = 2;
    
    % Compute pairwise alignment distances and best alignment shifts.
    alignmentDist = NaN(num_laps, num_laps);
    bestShift = NaN(num_laps, num_laps);
    for i = 1:num_laps
        for j = i+1:num_laps
            [d, delta] = alignment_distance(H_matrix(i, :), H_matrix(j, :), l_max, epsilon, p);
            alignmentDist(i, j) = d;
            alignmentDist(j, i) = d;
            bestShift(i, j) = delta;
            bestShift(j, i) = -delta; % symmetric: shift for j relative to i is the negative
        end
    end
    
    % (Optional) Display the pairwise alignment distance matrix.
    figure;
    imagesc(alignmentDist);
    colorbar;
    title('Pairwise Alignment Distance Matrix');
    xlabel('Lap Index');
    ylabel('Lap Index');
    
    % Save alignment analysis results in the output structure.
    lap_data_alignment.H_matrix = H_matrix;
    lap_data_alignment.x_grid = x_grid;
    lap_data_alignment.alignmentDist = alignmentDist;
    lap_data_alignment.bestShift = bestShift;
    
    % Optionally, you might want to return lap_data_alignment as part of your output.
    % For example:
    % lap_data.alignment = lap_data_alignment;
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Nested function: alignment_distance
function [D, best_delta] = alignment_distance(R1, R2, l_max, epsilon, p)
% alignment_distance computes the alignment distance between two
% vectors R1 and R2 (both defined on the same fixed grid) by shifting R1
% relative to R2. It returns the minimum distance D and the corresponding
% best shift (best_delta) in terms of bin indices.
%
% Inputs:
%   R1, R2   : Row vectors (of equal length) representing the interpolated H values.
%   l_max    : Maximum allowed shift (in indices).
%   epsilon  : Penalty coefficient for the shift.
%   p        : Exponent for the L^p norm (e.g., 2 for Euclidean).
%
% Outputs:
%   D         : The minimal distance achieved.
%   best_delta: The shift (in indices) that minimizes the distance.

N = length(R1);
best_distance = Inf;
best_delta = 0;
for delta = -l_max:l_max
    if delta < 0
        % Shift R1 to the right relative to R2.
        idx1 = (1 - delta):N;
        idx2 = 1:(N + delta);
    elseif delta > 0
        % Shift R1 to the left relative to R2.
        idx1 = 1:(N - delta);
        idx2 = (1 + delta):N;
    else
        idx1 = 1:N;
        idx2 = 1:N;
    end
    
    if isempty(idx1) || isempty(idx2)
        continue;
    end
    
    % Compute the L^p difference over the overlapping region.
    diff_vals = abs(R1(idx1) - R2(idx2)).^p;
    mean_diff = mean(diff_vals);
    
    % Add the shift penalty.
    current_distance = mean_diff + epsilon * abs(delta);
    
    if current_distance < best_distance
        best_distance = current_distance;
        best_delta = delta;
    end
end
D = best_distance;
end
