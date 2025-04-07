close all;
% Define the base directory
base_dir = '/Users/devenshidfar/Desktop/raw_tetrode_data/rat_913/';

% Define the file prefix and extension
file_prefix = 'm1_tt_';
file_ext = '.mat';

% Define the range of files to load
file_nums = 1:19;  % Adjust this

% Initialize an empty struct to hold all loaded structs
allData = struct();

% Loop through the file numbers
for i = file_nums
    % Construct the full filename
    file_name = sprintf('%s%02d%s', file_prefix, i, file_ext);
    file_path = fullfile(base_dir, file_name);
    
    % Check if the file exists
    if exist(file_path, 'file')
        % Load the .mat file
        data = load(file_path);
        
        % Extract the first struct inside the .mat file (assuming it's the main data)
        fieldNames = fieldnames(data);
        s = data.(fieldNames{1});  % Extract the struct
        
        % Store it in a new field in the main struct
        struct_name = sprintf('trial_%02d', i);
        allData.(struct_name) = s;
        
        fprintf('[INFO] Loaded: %s\n', file_path);
    else
        fprintf('[WARNING] File not found: %s\n', file_path);
    end
end

% Save all structs into one .mat file
output_file = fullfile(base_dir, 'all_tetrode_data.mat');
save(output_file, '-struct', 'allData');

fprintf('[INFO] Saved all tetrode structs to: %s\n', output_file);
