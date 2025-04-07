% =========  MAIN ENTRY  ==================================================
processAllSessions(full_trial_1, full_trial_2);     % <‑‑ call this

% =========  FUNCTIONS  ===================================================
function processAllSessions(full_trial_1, full_trial_2)
    % plot every session in both structs
    processDataset(full_trial_2.sessions,'full_trial_1');
    processDataset(full_trial_1.sessions,'full_trial_2');
end

function processDataset(sessions,tag)
    for k = 1:numel(sessions)
        processSessionStatic(sessions{k},sessions{k}.session_idx,tag);
    end
end

% ------------------------------------------------------------------------
function processSessionStatic(s,idx,tag)
% Static 3‑D plot of embeddings
%   • high‑velocity points coloured by hippocampal angle (jet)
%   • low‑velocity points over‑plotted in magenta

    % ------------ data ---------------------------------------------------
    E  = squeeze(s.embeddings_3d);          % Nx3
    H  = mod(s.binned_hipp_angle,2*pi);     % rad 0–2π
    V  = s.binned_vel;                      % deg / s

    % keep common length & finite rows
    n  = min([size(E,1) numel(H) numel(V)]);
    E  = E(1:n,:);   H = H(1:n);   V = V(1:n);
    good = all(isfinite(E),2) & isfinite(H) & isfinite(V);
    E = E(good,:);   H = H(good);   V = V(good);

    if isempty(E), warning('Session %d has no valid data',idx); return; end

    % velocity masks
    velThresh   = 5;                     % change if desired
    lowVelMask  = V <= velThresh;
    highVelMask = ~lowVelMask;

    % ------------ figure -------------------------------------------------
    fig = figure('Color','w','Position',[100 100 900 750]);
    ax  = axes('Parent',fig,'Projection','perspective');
    hold(ax,'on'); grid(ax,'on'); view(ax,3); axis(ax,'equal');

    % high‑velocity points coloured by hippocampal angle
    scatter3(ax,E(highVelMask,1),E(highVelMask,2),E(highVelMask,3),...
             18,H(highVelMask),'filled','MarkerFaceAlpha',0.75);

    % low‑velocity points in magenta
    scatter3(ax,E(lowVelMask,1),E(lowVelMask,2),E(lowVelMask,3),...
             28,'m','filled','MarkerFaceAlpha',0.85,'DisplayName','Low Vel');

    % colourbar & labels
    colormap(ax,'jet');  caxis(ax,[0 2*pi]);
    cb = colorbar(ax);   cb.Label.String = 'Hipp Angle (rad)';
    title(ax,sprintf('3‑D Embeddings  –  Session %d (%s)',idx,tag));
    xlabel(ax,'Dim 1');  ylabel(ax,'Dim 2');  zlabel(ax,'Dim 3');
    legend(ax,'show','Location','best');

    % ------------ save ---------------------------------------------------
    saveDir = fullfile('embeddings_plots',tag);
    if ~exist(saveDir,'dir'), mkdir(saveDir); end
    saveas(fig,fullfile(saveDir,sprintf('embeddings3D_session_%d.png',idx)));
    close(fig);
end
