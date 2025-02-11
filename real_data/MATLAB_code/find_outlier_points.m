%This file is to plot the landmark on and landmark off points on the same
%plot and see if and how they differ

n = 0.01

[sorted_dist_to_spline, sort_idx] = sort(full_trial_1.sessions{8}.dist_to_spline,'descend');
sorted_vel = full_trial_1.sessions{8}.binned_high_vel(sort_idx);

largest_outliers = sorted_dist_to_spline(1:n*length(sorted_dist_to_spline));

sorted_vel = sorted_vel(1:n*length(sorted_vel));

%plot(1:length(sorted_vel),sorted_vel,1:length(sorted_vel),largest_outliers)

plot(1:length(sorted_vel),sorted_vel);

mean_outlier_vel = mean(sorted_vel)

mean_vel = mean(full_trial_1.sessions{8}.binned_high_vel)

