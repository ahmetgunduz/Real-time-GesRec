import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import torch.onnx
import onnx
from onnx import optimizer
import os

import tensorrt as trt
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda 
import time


from utils import AverageMeter, calculate_accuracy
from models import squeezenet, shufflenetv2, shufflenet, mobilenet, mobilenetv2, c3d, resnetl


model_folder = 'results'
model_name = 'onnx_model'
onnx_model = 'results/ortho_model_shufflenetv2_unet_1120x224_best2.onnx'
# onnx_model = os.path.join(model_folder, model_name + '.onnx')


# # model = shufflenet.get_model(groups=3, width_mult=0.5, num_classes=600)#1
# # model = shufflenetv2.get_model( width_mult=0.25, num_classes=600, sample_size = 112)#2
# # model = mobilenet.get_model( width_mult=0.5, num_classes=600, sample_size = 112)#3
# # model = mobilenetv2.get_model( width_mult=0.2, num_classes=600, sample_size = 112)#4
# # model = shufflenet.get_model(groups=3, width_mult=1.0, num_classes=600)#5
# # model = shufflenetv2.get_model( width_mult=1.0, num_classes=600, sample_size = 112)#6
# # model = mobilenet.get_model( width_mult=1.0, num_classes=600, sample_size = 112)#7
# # model = mobilenetv2.get_model( width_mult=0.45, num_classes=600, sample_size = 112)#8
# # model = shufflenet.get_model(groups=3, width_mult=1.5, num_classes=600)#9
# # model = shufflenetv2.get_model( width_mult=1.5, num_classes=600, sample_size = 112)#10
# # model = mobilenet.get_model( width_mult=1.5, num_classes=600, sample_size = 112)#11
# # model = mobilenetv2.get_model( width_mult=0.7, num_classes=600, sample_size = 112)#12
# # model = shufflenet.get_model(groups=3, width_mult=2.0, num_classes=600)#13
# model = shufflenetv2.get_model( width_mult=2.0, num_classes=600, sample_size = 112)#14
# # model = mobilenet.get_model( width_mult=2.0, num_classes=600, sample_size = 112)#15
# # model = mobilenetv2.get_model( width_mult=1.0, num_classes=600, sample_size = 112)#16
# # model = squeezenet.get_model( version=1.1, num_classes=600, sample_size = 112, sample_duration = 8)
# # model = resnetl.resnetl10(num_classes=2, sample_size = 112, sample_duration = 8)
# # model = model.cuda()
# # model = nn.DataParallel(model, device_ids=None)  
# print(model)

# # create the imput placeholder for the model
# # input_placeholder = torch.randn(1, 3, 16, 112, 112)
# input_placeholder = torch.randn(1, 3, 1120, 224)


# # export
# torch.onnx.export(model, input_placeholder, onnx_model)
# print('{} exported!'.format(onnx_model))



TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(model_path):
    with trt.Builder(TRT_LOGGER) as builder, \
        builder.create_network() as network, \
        trt.OnnxParser(network, TRT_LOGGER) as parser: 
        builder.max_workspace_size = 1<<30
        builder.max_batch_size = 1
        with open(model_path, "rb") as f:
            parser.parse(f.read())
        engine = builder.build_cuda_engine(network)
        return engine

def alloc_buf(engine):
    # host cpu mem
    h_in_size = trt.volume(engine.get_binding_shape(0))
    h_out_size = trt.volume(engine.get_binding_shape(1))
    h_in_dtype = trt.nptype(engine.get_binding_dtype(0))
    h_out_dtype = trt.nptype(engine.get_binding_dtype(1))
    in_cpu = cuda.pagelocked_empty(h_in_size, h_in_dtype)
    out_cpu = cuda.pagelocked_empty(h_out_size, h_out_dtype)
    # allocate gpu mem
    in_gpu = cuda.mem_alloc(in_cpu.nbytes)
    out_gpu = cuda.mem_alloc(out_cpu.nbytes)
    stream = cuda.Stream()
    return in_cpu, out_cpu, in_gpu, out_gpu, stream


def inference(engine, context, inputs, out_cpu, in_gpu, out_gpu, stream):
    # async version
    # with engine.create_execution_context() as context:  # cost time to initialize
    # cuda.memcpy_htod_async(in_gpu, inputs, stream)
    # context.execute_async(1, [int(in_gpu), int(out_gpu)], stream.handle, None)
    # cuda.memcpy_dtoh_async(out_cpu, out_gpu, stream)
    # stream.synchronize()

    # sync version
    cuda.memcpy_htod(in_gpu, inputs)
    context.execute(1, [int(in_gpu), int(out_gpu)])
    cuda.memcpy_dtoh(out_cpu, out_gpu)
    return out_cpu

if __name__ == "__main__":
    # inputs = np.random.random((1, 3, 16, 112, 112)).astype(np.float32)
    inputs = np.random.random((1, 3, 1120, 224)).astype(np.float32)
    engine = build_engine(onnx_model)
    context = engine.create_execution_context()
    for _ in range(10):
        t1 = time.time()
        in_cpu, out_cpu, in_gpu, out_gpu, stream = alloc_buf(engine)
        res = inference(engine, context, inputs.reshape(-1), out_cpu, in_gpu, out_gpu, stream)
        print(res)
        print("cost time: ", time.time()-t1)


# tensorrt docker image: docker pull nvcr.io/nvidia/tensorrt:19.09-py3 (See: https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt/tags)
# NOTE: cuda driver >= 418





# batch_time = AverageMeter()
# input_var = Variable(torch.randn(1, 3, 8, 112, 112).cuda())
# end_time = time.time()

# for i in range(10000):

#   output = model(input_var)
#   batch_time.update(time.time() - end_time)
#   end_time = time.time()
#   print("Current average time: ", batch_time.avg, "Speed (vps): ", 1 / (batch_time.avg / 1.0) )

# print("Average time for GPU: ", batch_time.avg, "Speed (vps): ", 1 / (batch_time.avg / 1.0))









# import time
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import optim
# from torch.autograd import Variable

# from utils import AverageMeter, calculate_accuracy
# from models import squeezenet, shufflenetv2, shufflenet, mobilenet, mobilenetv2, c3d, resnetl

# try:
#     from apex.parallel import DistributedDataParallel as DDP
#     from apex.fp16_utils import *
#     from apex import amp, optimizers
#     from apex.multi_tensor_apply import multi_tensor_applier
# except ImportError:
#     raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


# # model = shufflenet.get_model(groups=3, width_mult=0.5, num_classes=600)#1
# # model = shufflenetv2.get_model( width_mult=0.25, num_classes=600, sample_size = 112)#2
# # model = mobilenet.get_model( width_mult=0.5, num_classes=600, sample_size = 112)#3
# # model = mobilenetv2.get_model( width_mult=0.2, num_classes=600, sample_size = 112)#4
# # model = shufflenet.get_model(groups=3, width_mult=1.0, num_classes=600)#5
# # model = shufflenetv2.get_model( width_mult=1.0, num_classes=600, sample_size = 112)#6
# # model = mobilenet.get_model( width_mult=1.0, num_classes=600, sample_size = 112)#7
# # model = mobilenetv2.get_model( width_mult=0.45, num_classes=600, sample_size = 112)#8
# # model = shufflenet.get_model(groups=3, width_mult=1.5, num_classes=600)#9
# # model = shufflenetv2.get_model( width_mult=1.5, num_classes=600, sample_size = 112)#10
# # model = mobilenet.get_model( width_mult=1.5, num_classes=600, sample_size = 112)#11
# # model = mobilenetv2.get_model( width_mult=0.7, num_classes=600, sample_size = 112)#12
# # model = shufflenet.get_model(groups=3, width_mult=2.0, num_classes=600)#13
# # model = shufflenetv2.get_model( width_mult=2.0, num_classes=600, sample_size = 112)#14
# # model = mobilenet.get_model( width_mult=2.0, num_classes=600, sample_size = 112)#15
# # model = mobilenetv2.get_model( width_mult=1.0, num_classes=600, sample_size = 112)#16
# # model = squeezenet.get_model( version=1.1, num_classes=600, sample_size = 112, sample_duration = 8)
# model = resnetl.resnetl10(num_classes=2, sample_size = 112, sample_duration = 8)
# model = model.cuda()
# #model = nn.DataParallel(model, device_ids=None)	
# optimizer = optim.SGD(
#             model.parameters(),
#             lr=0.001)
# print(model)

# print("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))
# assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

# model, optimizer = amp.initialize(model, optimizer,
#                                       opt_level='O3',
#                                       keep_batchnorm_fp32=True,
#                                       loss_scale=None
#                                       )

# batch_time = AverageMeter()
# input_var = Variable(torch.randn(1, 3, 8, 112, 112).cuda())
# end_time = time.time()

# for i in range(10000):

# 	output = model(input_var)
# 	batch_time.update(time.time() - end_time)
# 	end_time = time.time()
# 	print("Current average time: ", batch_time.avg, "Speed (vps): ", 1 / (batch_time.avg / 1.0) )

# print("Average time for GPU: ", batch_time.avg, "Speed (vps): ", 1 / (batch_time.avg / 1.0))
