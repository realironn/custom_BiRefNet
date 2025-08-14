#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().system('git clone https://github.com/ZhengPeng7/BiRefNet.git')
# get_ipython().system('pip uninstall -q torchaudio torchdata torchtext onnx onnxruntime onnxscript -y')
# get_ipython().system('pip install -qr BiRefNet/requirements.txt')
# get_ipython().system('pip install -q gdown onnx onnxscript onnxruntime-gpu==1.18.1')


# In[2]:


get_ipython().run_line_magic('cd', 'BiRefNet/')


# In[3]:


import torch
weights_file = '../best_model_epoch_103.pth'  # https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet-general-bb_swin_v1_tiny-epoch_232.pth
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[4]:


weights_file


# In[5]:


get_ipython().run_line_magic('pwd', '')


# In[6]:


with open('./config.py') as fp:
    file_lines = fp.read()
if 'swin_v1_tiny' in weights_file:
    print('Set `swin_v1_tiny` as the backbone.')
    file_lines = file_lines.replace(
        '''
            'pvt_v2_b2', 'pvt_v2_b5',               # 9-bs10, 10-bs5
        ][6]
        ''',
        '''
            'pvt_v2_b2', 'pvt_v2_b5',               # 9-bs10, 10-bs5
        ][3]
        ''',
    )
    with open('./config.py', mode="w") as fp:
        fp.write(file_lines)
else:
    file_lines = file_lines.replace(
        '''
            'pvt_v2_b2', 'pvt_v2_b5',               # 9-bs10, 10-bs5
        ][3]
        ''',
        '''
            'pvt_v2_b2', 'pvt_v2_b5',               # 9-bs10, 10-bs5
        ][6]
        ''',
    )
    with open('./config.py', mode="w") as fp:
        fp.write(file_lines)


# In[7]:


from utils import check_state_dict
from models.birefnet import BiRefNet


birefnet = BiRefNet(bb_pretrained=False)
state_dict = torch.load('./{}'.format(weights_file), map_location=device, weights_only=True)
state_dict = check_state_dict(state_dict)
birefnet.load_state_dict(state_dict)

torch.set_float32_matmul_precision(['high', 'highest'][0])

birefnet.to(device)
_ = birefnet.eval()


# In[8]:


birefnet


# In[9]:


get_ipython().system('git clone https://github.com/masamitsu-murase/deform_conv2d_onnx_exporter')
get_ipython().run_line_magic('cp', 'deform_conv2d_onnx_exporter/src/deform_conv2d_onnx_exporter.py .')
get_ipython().system('rm -rf deform_conv2d_onnx_exporter')


# In[10]:


with open('deform_conv2d_onnx_exporter.py') as fp:
    file_lines = fp.read()

file_lines = file_lines.replace(
    "return sym_help._get_tensor_dim_size(tensor, dim)",
    '''
    tensor_dim_size = sym_help._get_tensor_dim_size(tensor, dim)
    if tensor_dim_size == None and (dim == 2 or dim == 3):
        import typing
        from torch import _C

        x_type = typing.cast(_C.TensorType, tensor.type())
        x_strides = x_type.strides()

        tensor_dim_size = x_strides[2] if dim == 3 else x_strides[1] // x_strides[2]
    elif tensor_dim_size == None and (dim == 0):
        import typing
        from torch import _C

        x_type = typing.cast(_C.TensorType, tensor.type())
        x_strides = x_type.strides()
        tensor_dim_size = x_strides[3]

    return tensor_dim_size
    ''',
)

with open('deform_conv2d_onnx_exporter.py', mode="w") as fp:
    fp.write(file_lines)


# In[11]:


from torchvision.ops.deform_conv import DeformConv2d
import deform_conv2d_onnx_exporter

# register deform_conv2d operator
deform_conv2d_onnx_exporter.register_deform_conv2d_onnx_op()

def convert_to_onnx(net, file_name='output.onnx', input_shape=(1024, 1024), device=device):
    input = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)

    input_layer_names = ['input_image']
    output_layer_names = ['output_image']

    torch.onnx.export(
        net,
        input,
        file_name,
        verbose=False,
        opset_version=17,
        input_names=input_layer_names,
        output_names=output_layer_names,
    )
convert_to_onnx(birefnet, weights_file.replace('.pth', '.onnx'), input_shape=(1024, 1024), device=device)


# In[12]:


get_ipython().system('gdown 1ViUBfJdP10Hfh8YmotZY1Ea-74_r5MVf')


# In[13]:


from PIL import Image
from torchvision import transforms


transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

imagepath = './onnx_test-2.png'
image = Image.open(imagepath)
image = image.convert("RGB") if image.mode != "RGB" else image
input_images = transform_image(image).unsqueeze(0).to(device)
input_images_numpy = input_images.cpu().numpy()


# In[14]:


weights_file.replace('.pth', '.onnx')


# In[15]:


import onnxruntime


providers = ['CPUExecutionProvider'] if device == 'cpu' else ['CUDAExecutionProvider']
onnx_session = onnxruntime.InferenceSession(
    weights_file.replace('.pth', '.onnx'),
    providers=providers
)
input_name = onnx_session.get_inputs()[0].name
print(onnxruntime.get_device(), onnx_session.get_providers())


# In[16]:


from time import time
import matplotlib.pyplot as plt

time_st = time()
pred_onnx = torch.tensor(
    onnx_session.run(None, {input_name: input_images_numpy if device == 'cpu' else input_images_numpy})[-1]
).squeeze(0).sigmoid().cpu()
print(time() - time_st)

plt.imshow(pred_onnx.squeeze(), cmap='gray'); plt.show()


# In[17]:


with torch.no_grad():
    preds = birefnet(input_images)[-1].sigmoid().cpu()
plt.imshow(preds.squeeze(), cmap='gray'); plt.show()


# In[18]:


diff = abs(preds - pred_onnx)
print('sum(diff):', diff.sum())
plt.imshow((diff).squeeze(), cmap='gray'); plt.show()


# In[19]:


import numpy as np
from IPython.display import display

scale_ratio = 1024 / max(image.size)
scaled_size = (int(image.size[0] * scale_ratio), int(image.size[1] * scale_ratio))
image_masked_onnx = image.resize((1024, 1024))
image_masked_onnx.putalpha(transforms.ToPILImage()(pred_onnx))
display(image_masked_onnx.resize(scaled_size))
image_masked = image.resize((1024, 1024))
image_masked.putalpha(transforms.ToPILImage()(preds.squeeze()))
display(image_masked.resize(scaled_size))


# In[20]:


get_ipython().run_cell_magic('timeit', '', 'with torch.no_grad():\n    preds = birefnet(input_images)[-1].sigmoid().cpu()\n')


# In[21]:


get_ipython().run_cell_magic('timeit', '', 'pred_onnx = torch.tensor(\n    onnx_session.run(None, {input_name: input_images_numpy})[-1]\n).squeeze(0).sigmoid().cpu()\n')


# In[ ]:




