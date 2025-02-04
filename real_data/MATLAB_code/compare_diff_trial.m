% Number of trials
num_trials = 4;

% Preallocate separate arrays for SI_scores
SI_scores_1 = [];
SI_scores_2 = [];
SI_scores_3 = [];
SI_scores_4 = [];


% Iterate over each full_trial struct
for i = 1:num_trials
    % Dynamically access the full_trial struct
    trial_name = sprintf('full_trial_%d', i);
    trial_data = eval(trial_name);  % Retrieve the struct dynamically
    
    % Get the number of sessions in the trial
    num_sessions = length(trial_data.sessions);
    
    % Preallocate an array for this trial
    SI_scores = NaN(1, num_sessions); % Assuming numerical data
    
    % Iterate over sessions
    for j = 1:num_sessions
        SI_scores(j) = trial_data.sessions{j}.mean_H_difference;
    end
    
    % Assign to the correct variable based on trial index
    switch i
        case 1
            SI_scores_1 = SI_scores;
        case 2
            SI_scores_2 = SI_scores;
        case 3
            SI_scores_3 = SI_scores;
        case 4
            SI_scores_4 = SI_scores;
    end
end

% Compute means
mean_SI_1 = mean(SI_scores_1, 'omitnan');
mean_SI_2 = mean(SI_scores_2, 'omitnan');
mean_SI_3 = mean(SI_scores_3, 'omitnan');
mean_SI_4 = mean(SI_scores_4, 'omitnan');

% Display results
disp('Mean SI_score_hipp values for each trial:');
fprintf('Trial 1: %.4f\n', mean_SI_1);
fprintf('Trial 2: %.4f\n', mean_SI_2);
fprintf('Trial 3: %.4f\n', mean_SI_3);
fprintf('Trial 4: %.4f\n', mean_SI_4);

% Define session indices (assuming all trials have the same number of sessions)
num_sessions = length(SI_scores_1);
session_indices = 1:num_sessions;

% Create a figure
figure;
hold on;

% Plot each trial's SI_scores on the same y-axis
plot(session_indices+28, SI_scores_1, '-o', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Trial 1');
plot(session_indices+28, SI_scores_2, '-s', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Trial 2');
plot(session_indices+28, SI_scores_3, '-d', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Trial 3');
plot(session_indices+28, SI_scores_4, '-^', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Trial 4');

% Formatting the plot
xlabel('Session Index');
ylabel('mean H error');
title('SI Score Across Sessions for Each Trial');
legend('Location', 'best'); % Add a legend to distinguish trials
grid on;
hold off;
