import numpy as np
import scipy.io
import cebra
from scipy import stats
import matplotlib.pyplot as plt
import sys

# Load data
file_path = '/Users/devenshidfar/Desktop/Masters/control_recal/data_control_recal/dataverse_files/data/NN_opticflow_dataset.mat'
data = scipy.io.loadmat(file_path, squeeze_me=True, struct_as_record=False)
expt = data['expt']

print(expt.shape)

print(f"Experiment data type: {expt.dtype}")
print(f"Experiment data shape: {expt.shape}")

control_count = 0

for session_idx, session in enumerate(expt):

    control_count += 1

    if control_count <= 10:
        print(f"Skipping session {session_idx + 1}")
        continue 

    print(f"Processing session {session_idx + 1}/{len(expt)}")
    print(f"Rat: {session.rat}, Day: {session.day}, Epoch: {session.epoch}")

    # Prepare data
    ros_data = session.rosdata
    start_time = ros_data.startTs
    end_time = ros_data.stopTs
    bin_size = 0.025  # in seconds

    angle = ros_data.encAngle
    angle_times = ros_data.encTimes

    all_spikes = []
    num = 0
    for cluster in session.clust:
        try:
            spike_times = cluster.ts - start_time  # subtraction in microseconds
            print(f"Processing cluster: {cluster.name}")
            print(f"Number of spikes: {len(spike_times)}")
            
            # Bin spikes
            spike_times_sec = spike_times / 1e6
            start_time_sec = 0
            end_time_sec = (end_time - start_time) / 1e6
            
            if len(spike_times_sec) == 0:
                print("Warning: Empty spike_times array")
                continue  # Skip this cluster

            spike_times_sec = spike_times_sec[np.isfinite(spike_times_sec)]
            spike_times_sec = spike_times_sec[(spike_times_sec >= start_time_sec) & (spike_times_sec <= end_time_sec)]

            bins = np.arange(start_time_sec, end_time_sec + bin_size, bin_size)
            #print(f"bins: {bins[:5]}")

            binned_spikes, _, _ = stats.binned_statistic(
                spike_times_sec, 
                np.ones_like(spike_times_sec), 
                statistic='sum', 
                bins=bins
            )

            all_spikes.append(binned_spikes)
            #print(binned_spikes[20:35])
            print(f"Binned spikes shape: {binned_spikes.shape}")

        except Exception as cluster_error:
            print(f"Error processing cluster {cluster.name}: {str(cluster_error)}")
            continue

    print(f"Number of processed clusters: {len(all_spikes)}")

    if not all_spikes:
        print("No valid spike data found")
        continue

    print(f"all spikes shape {len(all_spikes)}")

    neural_data = np.array(all_spikes).T
    neural_times = np.arange(0, (end_time - start_time) / 1e6, bin_size)
    interp_angle = np.interp(neural_times, (angle_times - start_time) / 1e6, angle)

    print(f"Neural data shape: {neural_data.shape}")


    if not all_spikes:
        print("No valid spike data found")
        continue

    print(f"Neural data shape: {neural_data.shape}")

    # Apply CEBRA
    print(f"Input neural data shape: {neural_data.shape}")
    model = cebra.CEBRA(output_dimension=2, max_iterations=1000, batch_size=128)
    model.fit(neural_data)
    embeddings = model.transform(neural_data)
    print(f"Output embeddings shape: {embeddings.shape}")

    #model_shuffled = cebra.CEBRA(output_dimension=2, batch_size=128)
    #model_shuffled.fit(np.random.permutation(neural_data))
    #shuffled_embeddings = model_shuffled.transform(neural_data)

    # Visualize results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1])
    plt.colorbar(scatter, label='Angular Position (radians)')
    plt.title(f"Rat {session.rat}, Day {session.day}, Epoch {session.epoch} Embeddings")
    plt.xlabel('Embedding Dimension 1')
    plt.ylabel('Embedding Dimension 2')
    plt.show()

    # Optional: Save embeddings
    np.save(f'/Users/devenshidfar/Desktop/Masters/NRSC_510B/cebra_control_recal/angle_embeddings_discovery/embeddings_rat{session.rat}_day{session.day}_epoch{session.epoch}.npy', embeddings)

    # Break after the first session for debugging
    #break

print("Pipeline completed.")