
SI_unclust = zeros(1, 15);
SI_clust = zeros(1, 15);

% Extract values
for i = 1:15
    SI_unclust(i) = full_trial_1(1).sessions{i}.SI_score_hipp;
    SI_clust(i) = full_trial_2(1).sessions{i}.SI_score_hipp;
end

% Plot the difference across sessions
figure;
plot(1:15, SI_clust - SI_unclust, '-o');
xlabel('Session');
ylabel('ΔSI (Clust - Unclust)');
title('Difference in SI\_score\_hipp (Clustered vs. Unclustered)');
grid on;