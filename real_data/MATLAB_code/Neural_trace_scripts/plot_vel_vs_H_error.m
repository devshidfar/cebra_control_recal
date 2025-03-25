close all;

H_error = [];
prcnt_vels = [];
prcnt_neural_low = [];
session_indices = [];

neural_threshold = 0.05;

for i = 1:30
    % Store H error and session indices
    H_error = [H_error, full_trial_1.sessions{i}.mean_H_difference];
    session_idx = full_trial_1.sessions{i}.session_idx;
    session_indices = [session_indices, session_idx];

    % Neural data analysis
    neural_data = full_trial_1.sessions{i}.neural_data_full_trial; % neurons x time array
    avg_neural_activity = mean(neural_data, 1); % average across neurons at each time point
    neural_low_fraction = sum(avg_neural_activity < neural_threshold) / length(avg_neural_activity) * 100;
    prcnt_neural_low = [prcnt_neural_low, neural_low_fraction];

    % Velocity analysis
    vels = full_trial_1.sessions{i}.binned_vel;
    length_low_vel = length(find(vels < 5));
    prcnt_vels = [prcnt_vels, (length_low_vel / length(vels)) * 100];
end

% Scatter plot using neural activity low percentage
scatter(prcnt_neural_low, H_error, 'b', 'filled');
xlabel('% Trial Avg Neural Activity < Threshold');
ylabel('H Error');
title('Scatter Plot of % Low Neural Activity vs. H Error');
grid on;

% Annotate each point with session index
for i = 1:length(prcnt_neural_low)
    text(prcnt_neural_low(i), H_error(i), num2str(session_indices(i)), ...
        'FontSize', 10, 'Color', 'red', ...
        'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
end
