
close all
ranges = nan(16, 2);  % preallocate for 16 sessions (36 to 51), columns: [min, max]

for sessionIdx = 36:51
    relativeIdx = sessionIdx - 35;  % since you're storing in rows 1–16

    if ~isfield(expt(sessionIdx), 'unclustered') || isempty(expt(sessionIdx).unclustered)
        fprintf('Session %d has no unclustered data.\n', sessionIdx);
        continue;
    end

    unclustered_data_struct = expt(sessionIdx).unclustered;
    start_ts = expt(sessionIdx).rosdata.startTs;
    stop_ts = expt(sessionIdx).rosdata.stopTs / (1e6 * 60);

    all_ts = [];

    for i = 1:numel(unclustered_data_struct)
        ts_i = unclustered_data_struct(i).ts(:);
        all_ts = [all_ts; ts_i];
    end

    all_ts_min =(all_ts -start_ts) / (1e6 * 60); % Make in min

    % Save min and max
    ranges(relativeIdx, :) = [min(all_ts_min), max(all_ts_min)];

    % Plot
    figure;
    histogram(all_ts_min, 30);
    title(sprintf('Combined Spike Times — Session %d - Stop Time %.2f', sessionIdx,stop_ts));
    xlabel('Time (min)');
    ylabel('Spike Count');
end
