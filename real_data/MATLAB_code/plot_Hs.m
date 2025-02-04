function plotHs(decode_H, binned_est_gain, session_idx, dataset_name)
    % Ensure input arrays are not empty
    if isempty(decode_H) || isempty(binned_est_gain)
        error('Input arrays decode_H and binned_est_gain must not be empty.');
    end

    % Create X-axis (index values)
    x = 1:length(decode_H);

    % Create figure
    fig = figure('Visible', 'off'); % Don't display the figure
    hold on;

    % Plot decode_H in blue
    plot(x, decode_H, 'b-', 'LineWidth', 1.5);
    
    % Plot binned_est_gain in red
    plot(x, binned_est_gain, 'r-', 'LineWidth', 1.5);
    
    hold off;

    % Add labels, legend, and title
    xlabel('Index');
    ylabel('Value');
    title(sprintf('%s: Manifold Decoded H vs Fourier H (Session %d)', dataset_name, session_idx));
    legend('decode H', 'binned est gain', 'Location', 'best');
    grid on;

    % Create directory for saving plots
    output_folder = 'decoded_H_vs_binned_gain_plots';
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end

    % Save the figure
    save_path = fullfile(output_folder, sprintf('%s_session_%d.png', dataset_name, session_idx));
    saveas(fig, save_path);

    % Close the figure to free memory
    close(fig);
end


plotHs(full_trial_1.sessions{8}.lap_decode_H, full_trial_1.sessions{8}.lap_est_gain, full_trial_1.sessions{8}.session_idx, 'Full Trial');