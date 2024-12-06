# README

This README provides an overview my project, its purpose, the methods and libraries it relies on, and guidelines for running and reproducing the analysis. The code is designed for analyzing spike rate neural data, generating low-dimensional embeddings using CEBRA, fitting principal curves, and performing decoding and spectral analyses to understand neural manifolds in rat navigation tasks.

**Main Code File** is real_data/run_embed_calc_H.ipynb <br />
Helper functions file is in cebra_utils.py  <br />
Helper functions from Chaudhuri et al., is in real_data/Shared_Scripts/Spud_Code  <br />
Helper functions from Sebastien et al., is in real_data/SI_code

## Overview of the Code

The code performs the following key steps:

1. **Data Loading and Preprocessing:**  
   - Extracts hippocampal neural spike data, behavioural variables (e.g., angle, velocity), and session metadata.
   - Filters and bins the data in 1 sedcond bins and removes time points with running speed less than 5 degrees per second (variable).
   - Removes clusters with isolation quality higher than 4 (see Madhav et al., 2019).

2. **Dimensionality Reduction and Manifold Learning (CEBRA):**  
   - Uses CEBRA (https://cebra.ai/) to find low-dimensional embeddings of the high-dimensional neural spike data.
   - Reveals latent geometric structures (“manifolds”) underlying the neural population activity.

3. **Outlier Removal and Curve Fitting:**  
   - Identifies and removes outliers from the embeddings for cleaner manifold visualization and proper further analysis.
   - Fits a principal curve to the low-dimensional manifold (inspired by Chaudhuri et al., 2019), providing a parameterization of the neural representation.

4. **Behavioral Variable Decoding and Structure Index:**  
   - Decodes “hippocampal angle" from spline and then calculates “hippocampal gain” from the embeddings. see (Jayakumar et al., 2019, Madhav et al., 2024) for descriptions of hippocampal angle and hippocampal gain.
   - Computes a Structure Index (SI) to quantify manifold quality, see (https://doi.org/10.1101/2022.11.23.517657).
   - Performs spatial spectrograms of resulting "hippocampal gain" values to identify dominant frequencies in the Hippocampal gain to determine within lap patterns.

5. **Visualization and Saving Results:**  
   - Generates static and interactive plots of embeddings, principal curves, and decoded variables.
   - Saves figures, CSV summaries, and embeddings to files.
   - Creates a .pdf file for analyses
   - Creates PDF outputs for compiled figures and parameters.
   - Handles multiple sessions and conditions.

## Dependencies

- **Python 3.9.19**
- **NumPy (e.g., 1.24.3), SciPy (1.10.1)** for computations and stats
- **Matplotlib (3.7.1)** for plotting
- **CEBRA (0.1.1)** for neural embedding
- **scikit-learn, umap, ripser** for manifold analysis and persistent homology
- **pandas** for general data manipulation
- Additional packages (e.g., mpld3, plotly) for interactive visualization

Install these using `pip` or `conda` as needed.

## Data Requirements

  
- **Data Format:**  
  Ensure your data match the expected structure. You may have to change data loading fields and methods to have it work for your specific data.

## Running the Code

1. **Set Up Your Environment:**  
   - Create a virtual environment (recommended).
   - Install the required packages.

2. **Prepare Data:**  
   - Load your dataset and ensure the necessary session objects are defined.
   - Check and set file paths for saving models and results.

3. **Configure Parameters:**  
   - Adjust bin sizes, velocity thresholds, and other parameters in the code.
   - Modify save paths and flags (`save_models`, `save_anim`, `load_npy`) as needed.

4. **Run the Script:**  
   - Execute the script in your preferred environment.
   - The code processes each session, generates embeddings, fits principal curves, computes SI scores, decodes variables, and saves outputs.

## Output

- **Embeddings and Curves:**  
  Saved as `.npy` or `.mat` files.

- **Figures and Animations:**  
  `.png`, `.pdf`, or `.gif` showing embeddings, decoded variables, and spectrograms.

- **CSV Files:**  
  Summaries of SI scores, decoding errors, isolation quality metrics, etc.

## Notes and Caveats

- The code is tailored to a specific dataset and may require adaptation for others.
- Some functions (e.g., `apply_cebra()`, `fit_spud_to_cebra()`) rely on custom or external modules.
- Interactive plots may require a GUI or browser environment.

## Further Information

Refer to the cited literature 

(Madhav et al., 2024; Madhav et al., 2022; Jayakumar et al., 2019; Chaudhuri et al., 2019; Schneider et al., 2023) for theoretical grounding and methodological details.

Chaudhuri, R., Gerc ̧ek, B., Pandey, B., Peyrache, A., & Fiete, I. (2019). The intrinsic attractor manifold and population dynamics of a canonical cognitive circuit across waking and sleep. Nature Neuro- science, 22(9), 1512–1520. https://doi.org/10.1038/s41593-019-0460-x

Jayakumar, R. P., Madhav, M. S., Savelli, F., Blair, H. T., Cowan, N. J., & Knierim, J. J. (2019). Re- calibration of path integration in hippocampal place cells. Nature, 566(7745), 533–537. https: //doi.org/10.1038/s41586-019-0939-3

Madhav, M. S., Jayakumar, R. P., Lashkari, S. G., Savelli, F., Blair, H. T., Knierim, J. J., & Cowan, N. J. (2022). The Dome: A virtual reality apparatus for freely locomoting rodents. Journal of Neuroscience Methods, 368, 109336. https://doi.org/10.1016/j.jneumeth.2021.109336

Madhav, M. S., Jayakumar, R. P., Li, B. Y., Lashkari, S. G., Wright, K., Savelli, F., Knierim, J. J., & Cowan, N. J. (2024). Control and recalibration of path integration in place cells using optic flow [Publisher: Nature Publishing Group]. Nature Neuroscience, 27(8), 1599–1608. https://doi.org/10. 1038/s41593-024-01681-9

Schneider, S., Lee, J. H., & Mathis, M. W. (2023). Learnable latent embeddings for joint behavioural and neural analysis. Nature, 617(7960), 360–368. https://doi.org/10.1038/s41586-023-06031-6

## Contact

For further assistance or questions regarding the code and its dependencies, please contact:

- **Deven Shidfar**  
  Email: [devenshidfar@math.ubc.ca](mailto:devenshidfar@math.ubc.ca)  
  Affiliation: University of British Columbia 

Feel free to reach out for support, collaboration opportunities, or any inquiries related to this project. Additionally, refer to the project documentation for more detailed information on methodologies and experimental setups.

---

This README helps you understand the code’s purpose, setup, execution, and output. Adjust as necessary for your specific data and environment.

