import torch
import os
import os.path as path

dict_size = 0
data_dir = 'data/'
v3_dir = path.join(data_dir, 'codes_v3')
v2_dir = path.join(data_dir, 'codes')

for dirname, _, filenames in os.walk(v3_dir):
    for filename in filenames:
        v3_filepath = path.join(dirname, filename)
        v3_data = torch.load(v3_filepath, map_location=torch.device('cpu'))
        dict_size = max(dict_size, torch.max(v3_data).item())
        v2_filepath = path.join(v2_dir, filename)
        torch.save(v3_data, v2_filepath, _use_new_zipfile_serialization=False)

print(f'Number of embeddings: {dict_size}')
