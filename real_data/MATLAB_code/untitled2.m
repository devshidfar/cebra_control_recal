% Define the window size for moving average
window_size = 50;  % Adjust this based on how much smoothing you want

% Compute the moving average
smoothed_vel = movmean(struct.binned_vel, window_size);

% Plot original and smoothed data
figure;
plot(1:length(struct.binned_vel), struct.binned_vel, 'b', 'DisplayName', 'Original Velocity');
hold on;
plot(1:length(struct.binned_vel), smoothed_vel, 'r', 'LineWidth', 2, 'DisplayName', 'Smoothed Velocity');
hold off;

% Labels and legend
xlabel('Time');
ylabel('Velocity');
title('Velocity with Moving Average Smoothing');
legend;
grid on;
