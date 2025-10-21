import pickle
import torch
from torch.utils.data import Dataset
from utils import MackeyGlass
# Assuming the MackeyGlass class and dependencies (jitcdde_lyap, numpy) are available

# 1. Define the parameters for the Mackey-Glass series
mg_params = {
    'tau': 17,
    'constant_past': 0.9,
    'dt': 1.0,
    'splits': (10000., 0.), 
    'seed_id': 42
}

# Instantiate the class to generate the data
print("Generating Mackey-Glass time series...")
mg_dataset = MackeyGlass(**mg_params)

time_series_list = []

# Iterate over all available indices (0 to len-1) and collect the targets (x_{t+1}).
# The loop runs for 10,000 steps, collecting x_0 through x_10000.
for idx in range(len(mg_dataset)):
    # __getitem__ returns (sample, target). We only need the target here.
    _, target = mg_dataset[idx] 
    time_series_list.append(target.squeeze().item())

# Assemble the Final Two-Column Tensor
# The collected list now contains the full time series (e.g., 10001 points).

# Convert the list to a column tensor (Shape: N x 1)
series_column_1 = torch.tensor(time_series_list, dtype=torch.float64).unsqueeze(1)

# Create the second column by duplicating the first
series_column_2 = series_column_1.clone()

# Concatenate the two columns to form the N x 2 output tensor
# Column 1: Mackey-Glass series (x_0, x_1, x_2, ...)
# Column 2: Mackey-Glass series (x_0, x_1, x_2, ...) for later forecasting use
time_series_data = torch.cat((series_column_1, series_column_2), dim=1)

print(f"Generation complete. Data shape: {time_series_data.shape}")

# Define the output filename
output_filename = "data.pkl"

# Save the tensor to a .pkl file
with open(output_filename, 'wb') as f:
    pickle.dump(time_series_data, f)

print(f"Successfully saved the two-column time series to '{output_filename}'")

# Verification
with open(output_filename, 'rb') as f:
    loaded_data = pickle.load(f)
print(f"Verification: Loaded data shape is {loaded_data.shape}")
print(f"Verification: First few points of Column 1: {loaded_data[:5, 0].tolist()}")
print(f"Verification: First few points of Column 2: {loaded_data[:5, 1].tolist()}")
