% Assume your data is stored in a matrix called 'activity'
% Rows = timepoints (392), columns = neurons (46)
% Example: activity = rand(392, 46);  % for testing

mean_activity = smooth(mean(full_trial_2.sessions{3}.neural_data_low_vel, 2))  % average across neurons (columns)

figure;
plot(mean_activity, 'LineWidth', 1.5);
xlabel('Time');
ylabel('Mean Activity');
title('Mean Activity Over Time');
grid on;
