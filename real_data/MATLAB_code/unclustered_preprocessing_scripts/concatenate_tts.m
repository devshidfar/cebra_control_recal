%% Load expt if needed
expt_data = load('/Users/devenshidfar/Desktop/Masters/NRSC_510B/cebra_control_recal/mat_code_and_data/data/NN_opticflow_dataset.mat');
expt = expt_data.expt; 
% % or expt = expt_data.whateverField; 
% % Make sure expt is indeed a 1x65 struct.

%% 1) Get a list of .mat files (m1_tt_*.mat).
fileList = dir('/Users/devenshidfar/Desktop/raw_tetrode_data/rat_913/m1_tt_*.mat');  
nFiles   = numel(fileList);

%% 2) Loop through each .mat file.
for iFile = 1:nFiles
    fname = fileList(iFile).name;
    
    % Extract day and rat from filename: m1_tt_{day}_{rat}.mat
    tokens = regexp(fname, '^m1_tt_(\d+)_(\d+)\.mat$', 'tokens');
    if isempty(tokens)
        warning('File %s does not match pattern m1_tt_{day}_{rat}.mat', fname);
        continue;
    end
    tokens = tokens{1};   % e.g. {'14','913'}
    file_rat_str = tokens{1}; 
    file_day_str = tokens{2}; 
    
    % Convert these to numeric (assuming day and rat are numeric in expt).
    file_rat_num = str2double(file_rat_str);
    file_day_num = str2double(file_day_str);

    fprintf('Checking file: %s | Extracted day: %d | Extracted rat: %d\n', ...
        fname, file_day_num, file_rat_num);

    %% 3) Find the matching row in expt
    matchIdx = [];
    for j = 1:numel(expt)
        % If expt.day or expt.rat is stored differently, adjust here
        expt_day_num = expt(j).day;      % e.g., numeric
        expt_rat_num = expt(j).rat;      % e.g., numeric

        % Debugging output to see each comparison
        fprintf('  Expt entry %d: day = %d, rat = %d\n', j, expt_day_num, expt_rat_num);

        if expt_day_num == file_day_num && expt_rat_num == file_rat_num
            matchIdx = j;
            fprintf('  Match found at index %d\n', j);
            break;
        end
    end
    
    % If still empty, no match found
    if isempty(matchIdx)
        warning('No matching expt found for file %s (day=%d, rat=%d)', ...
            fname, file_day_num, file_rat_num);
        continue;
    end

    %% 4) Load the unclustered spike data from the file
    data = load(fullfile(fileList(iFile).folder, fname));
    varNames = fieldnames(data);
    unclusteredData = data.(varNames{1});  % Usually there's one variable

    % Extract only the first 4 fields: name, ttnum, ts, vel
    fieldsToKeep = {'name', 'ttnum', 'ts', 'vel'};
    filteredUnclustered = struct([]);

    for k = 1:numel(unclusteredData)
        % disp(k)
        for f = 1:numel(fieldsToKeep)
            if isfield(unclusteredData(k), fieldsToKeep{f})
                filteredUnclustered(k).(fieldsToKeep{f}) = ...
                    unclusteredData(k).(fieldsToKeep{f});
            end
        end
    end

    %% 5) Append the filtered unclustered data to the matched expt row
    expt(matchIdx).unclustered = filteredUnclustered;
    fprintf('Added filtered unclustered data from %s to expt index %d\n', fname, matchIdx);
end

%% 6) Put zeros everywhere for rows that aren't rat 913
for i = 1:numel(expt)
    % If rat is stored as numeric
    if expt(i).rat ~= 913
        fieldNames = fieldnames(expt(i));
        for f = 1:numel(fieldNames)
            fieldValue = expt(i).(fieldNames{f});
            
            if isnumeric(fieldValue)
                % If the field is numeric, set it to zero
                expt(i).(fieldNames{f}) = 0;
            elseif isstruct(fieldValue)
                % If it's a struct, replace with an empty struct
                expt(i).(fieldNames{f}) = struct();
            elseif iscell(fieldValue)
                % If it's a cell, replace with an empty cell
                expt(i).(fieldNames{f}) = {};
            else
                % Otherwise (string, char, etc.), replace with an empty string
                expt(i).(fieldNames{f}) = "";
            end
        end
    end
end

%% 7) Display final struct array
disp(expt);
