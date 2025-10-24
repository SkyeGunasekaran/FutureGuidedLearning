import pickle
import torch
from torch.utils.data import Dataset
<<<<<<< HEAD
from utils import MackeyGlass
=======
>>>>>>> 07c6981c24b9b0cf889dc0ebfcffe739062da438
# Assuming the MackeyGlass class and dependencies (jitcdde_lyap, numpy) are available

# Define the parameters for the Mackey-Glass series
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

<<<<<<< HEAD
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
=======
# These lists will store the two columns.
samples_list = []  # Column 0: [x_0, x_1, ..., x_{N-1}]
targets_list = []  # Column 1: [x_1, x_2, ..., x_N]

# Iterate from idx=0 up to len(mg_dataset) - 1.
# For splits=(10000., 0.), len() is 10000. Indices run from 0 to 9999.
for idx in range(len(mg_dataset)):
    # __getitem__(idx) returns (x_{idx}, x_{idx+1})
    sample, target = mg_dataset[idx] 
    
    # Store the current sample (x_t) for Column 0
    samples_list.append(sample.squeeze().item())
    
    # Store the target (x_{t+1}) for Column 1
    targets_list.append(target.squeeze().item())

# Convert lists to column tensors (Shape: N x 1)
series_column_0 = torch.tensor(samples_list, dtype=torch.float64).unsqueeze(1) # x_0 to x_{N-1}
series_column_1 = torch.tensor(targets_list, dtype=torch.float64).unsqueeze(1) # x_1 to x_N

# Concatenate the two columns (Shape: N x 2)
time_series_data = torch.cat((series_column_0, series_column_1), dim=1)

print(f"Generation complete. Data shape: {time_series_data.shape}")

output_filename = "data.pkl"

with open(output_filename, 'wb') as f:
    pickle.dump(time_series_data, f)

print(f"Successfully saved the lagged time series to '{output_filename}'")

# (Optional) Verification
with open(output_filename, 'rb') as f:
    loaded_data = pickle.load(f)

# Verification checks
print(f"Verification: Loaded data shape is {loaded_data.shape}")
print(f"Verification: First point of Column 0 (x_0): {loaded_data[0, 0].item()}")
print(f"Verification: First point of Column 1 (x_1): {loaded_data[0, 1].item()}")
print(f"Verification: x_1 (Col 0, index 1) == x_1 (Col 1, index 0): {loaded_data[1, 0].item() == loaded_data[0, 1].item()}")
>>>>>>> 07c6981c24b9b0cf889dc0ebfcffe739062da438
