import torch
path_name = "weights/egogesture_resnetl_10_Depth_8.pth"
path_name = "weights/egogesture_resnext_101_Depth_32.pth"
ckpt = torch.load(path_name, map_location='cpu', weights_only=False)
print(list(ckpt['state_dict'].keys())[-1])  # last layer name + shape
print(ckpt['state_dict'][list(ckpt['state_dict'].keys())[-1]].shape)