% Define window size for moving average
window_size = 50;
i = 21;
% Extract data
velocity = full_trial_1.sessions{i}.binned_vel;
mean_H_error = full_trial_1.sessions{i}.mean_H_difference;
session_idx = full_trial_1.sessions{i}.session_idx;

% Compute moving average
smoothed_vel = movmean(velocity, window_size);

% Plot original and smoothed data
figure;
plot(velocity, 'b', 'DisplayName', 'Original Velocity');
hold on;
plot(smoothed_vel, 'r', 'LineWidth', 2, 'DisplayName', 'Smoothed Velocity');

% Annotate mean_H_error and session_idx on plot (top-right corner)
x_pos = length(velocity) * 0.7; % Adjust as needed
y_pos = max(velocity) * 0.9;    % Adjust as needed

% Correct combined string
annotation_text = sprintf('Mean H Error = %.2f\nSession idx = %d', mean_H_error, session_idx);

text(x_pos, y_pos, annotation_text, ...
     'FontSize', 12, 'Color', 'k', 'FontWeight', 'bold', 'BackgroundColor','w');
hold off;

% Labels and legend
xlabel('Time');
ylabel('Velocity');
title('Velocity with Moving Average Smoothing');
legend('Location','best');
grid on;
