import pickle
import torch
from torch.utils.data import Dataset
from jitcdde import jitcdde_lyap
import numpy as np
from utils import MackeyGlass

# 1. Define the parameters for the Mackey-Glass series
mg_params = {
    'tau': 17,
    'constant_past': 0.9,
    'dt': 1.0,
    'splits': (10000., 0.), # 10,000 points split 60/20/20
    'seed_id': 42
}

# 2. Instantiate the class to generate the data
print("Generating Mackey-Glass time series...")
mg_dataset = MackeyGlass(**mg_params)

# The full time series is stored in the `mackeyglass_soln` attribute
time_series_data = mg_dataset.mackeyglass_soln
print(f"Generation complete. Data shape: {time_series_data.shape}")

# 3. Define the output filename
output_filename = "mackey_glass_series.pkl"

# 4. Save the tensor to a .pkl file
with open(output_filename, 'wb') as f:
    pickle.dump(time_series_data, f)

print(f"Successfully saved the time series to '{output_filename}'")

# (Optional) Verify by loading the data back
with open(output_filename, 'rb') as f:
    loaded_data = pickle.load(f)

print(f"Verification: Loaded data shape is {loaded_data.shape}")
assert torch.equal(time_series_data, loaded_data)
print("Verification successful!")
