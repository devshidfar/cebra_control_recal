
SI_unclust = zeros(1, 15);
SI_clust = zeros(1, 15);

% Extract values
for i = 1:15
    SI_unclust(i) = full_trial_1(1).sessions{i}.SI_score_hipp;
    SI_clust(i) = full_trial_2(1).sessions{i}.SI_score_hipp;
    H_error_unclust(i) = full_trial_1(1).sessions{i}.mean_H_difference;
    H_error_clust(i) = full_trial_2(1).sessions{i}.mean_H_difference;
end

session_range = 35:49;

% Plot both lines
% figure;
% plot(session_range, SI_clust, '-o', 'LineWidth', 2); hold on;
% plot(session_range, SI_unclust, '-o', 'LineWidth', 2);
% 
% xlabel('Session Index');
% ylabel('SI\_score\_hipp');
% title('Structure Index Comparison (Clustered vs. Unclustered)');
% legend('Clustered', 'Unclustered');
% grid on;
figure;
plot(session_range, H_error_clust, '-o', 'LineWidth', 2); hold on;
plot(session_range, H_error_unclust, '-o', 'LineWidth', 2);

xlabel('Session Index');
ylabel('SI\_score\_hipp');
title('Structure Index Comparison (Clustered vs. Unclustered)');
legend('Clustered', 'Unclustered');
grid on;
