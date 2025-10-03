#import torch
#print(torch.version.cuda)
#print(torch.cuda.is_available())


import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")