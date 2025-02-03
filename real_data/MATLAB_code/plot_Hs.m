figure;
x = 1:length(land_off.sessions{1}.decode_H)
plot(x, land_off.sessions{1}.decode_H, 'b-', 'LineWidth', 1.5); % Plot decode_H in blue
hold on;
plot(x, land_off.sessions{1}.binned_est_gain, 'r-', 'LineWidth', 1.5); % Plot binned_est_gain in red
hold off;

% Add labels, legend, and title
xlabel('Index');
ylabel('Value');
title('Land Off: Manifold Decoded H vs Fourier H');
legend('decode H', 'binned est gain');
grid on;