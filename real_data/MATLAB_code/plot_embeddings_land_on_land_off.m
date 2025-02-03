function plotTwo3DArrays(array1, array2)
    % Ensure arrays have the correct dimensions
    if size(array1, 2) ~= 3 || size(array2, 2) ~= 3
        error('Both arrays must have 3 columns for (X, Y, Z) coordinates.');
    end

    % Extract X, Y, Z coordinates for each array
    x1 = array1(:, 1); y1 = array1(:, 2); z1 = array1(:, 3);
    x2 = array2(:, 1); y2 = array2(:, 2); z2 = array2(:, 3);

    % Create figure
    figure;
    hold on;
    grid on;
    
    % Plot first array in red
    scatter3(x1, y1, z1, 50, 'r', 'filled');

    % Plot second array in blue
    scatter3(x2, y2, z2, 50, 'b', 'filled');

    % Labels and title
    xlabel('X Axis');
    ylabel('Y Axis');
    zlabel('Z Axis');
    title('3D Scatter Plot of Two Arrays');

    % Legend
    legend('Array 1 (Red)', 'Array 2 (Blue)', 'Location', 'best');

    % Adjust view
    view(3);
    axis equal;
    
    hold off;
end

plotTwo3DArrays(land_off.sessions{1}.embeddings_3d,no_land_off.sessions{2}.embeddings_3d)
