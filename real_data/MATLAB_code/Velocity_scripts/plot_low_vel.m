
close all;
struct = full_trial_1.sessions{12};

array1 = struct.embeddings_3d;
array2 = struct.embeddings_low_vel;


figure;
hold on; % Allows multiple plots on the same figure

% Scatter plot for the first array in red
scatter3(array1(:,1), array1(:,2), array1(:,3), 30, 'r', 'filled');

% Scatter plot for the second array in blue
scatter3(array2(:,1), array2(:,2), array2(:,3), 30, 'b', 'filled');

hold off;

% Labels and formatting
xlabel('X');
ylabel('Y');
zlabel('Z');
title('3D Scatter Plot of High vel and low Vel embeddings');
legend({'High Vel', 'Low Vel'}, 'Location', 'best');
grid on;
view(3); % Ensures a proper 3D view