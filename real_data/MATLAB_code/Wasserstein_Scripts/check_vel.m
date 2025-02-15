% spike_times = full_trial_1.sessions{1}.expt_file.rosdata.encTimes
% vel = full_trial_1.sessions{1}.expt_file.rosdata.vel
% 
% bin_size = 1
% 
% bins = spike_times(1):bin_size:spike_times(end) + bin_size;
% 
% 
% try
%     % Use histcounts to bin data and accumarray to compute mean per bin
%     [~, ~, binIdx] = histcounts(spike_times, bins);
% 
%     % Ensure binIdx is valid
%     validIdx = binIdx > 0;
% 
%     % Compute mean of est_gain_filtered per bin
%     binned_vel = accumarray(binIdx(validIdx), vel(validIdx), [], @mean, NaN);
% catch ME
%     disp(['[ERROR] Exception occurred: ', ME.message]);
% end

vel = full_trial_2.sessions{1}.binned_high_vel 
vel
count = sum(vel <= 5)
where = find(vel <= 5)

vel(41:43)
55
67
73