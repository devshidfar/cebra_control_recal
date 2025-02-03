H_error_land_on = [];
H_error_full = [];
H_error_train_land_on = [];

SI_score_land_on = [];
SI_score_full = [];
SI_score_train_land_on = [];

for i = 1:length(land_on.sessions)
    H_error_land_on(end + 1) = land_on.sessions{i}.mean_H_difference;
    H_error_full(end + 1) = full_trial.sessions{i}.mean_H_difference;
    H_error_train_land_on(end + 1) = train_land_on.sessions{i}.mean_H_difference;

    SI_score_land_on(end + 1) = land_on.sessions{i}.SI_score_hipp;
    SI_score_full(end + 1) = full_trial.sessions{i}.SI_score_hipp;
    SI_score_train_land_on(end + 1) = train_land_on.sessions{i}.SI_score_hipp;
end

H_error_land_on_filtered = H_error_land_on(SI_score_land_on > 0.75);
H_error_full_filtered = H_error_full(SI_score_full > 0.75);
H_error_train_land_on_filtered = H_error_train_land_on(SI_score_train_land_on > 0.75);

filtered_mean_H_error_land_on = mean(H_error_train_land_on_filtered, 'omitnan')
filtered_mean_H_error_land_full = mean(H_error_full_filtered, 'omitnan')
filtered_mean_H_error_train_land_on = mean(H_error_land_on_filtered, 'omitnan')


mean_H_error_land_on = mean(H_error_train_land_on, 'omitnan')
mean_H_error_land_full = mean(H_error_full, 'omitnan')
mean_H_error_train_land_on = mean(H_error_land_on, 'omitnan')

num_land_on = sum(SI_score_land_on > 0.75)
num_full_trial = sum(SI_score_full > 0.75)
num_train_land_on = sum(SI_score_train_land_on > 0.75)