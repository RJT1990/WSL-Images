from torchbench.image_classification import ImageNet
import torchvision.transforms as transforms
import PIL
import torch

# Define Transforms    
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
b0_input_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

# Run Evaluation
ImageNet.benchmark(
    model=torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x16d_wsl'),
    paper_model_name='ResNeXt-101 32x48d',
    paper_arxiv_id='1805.00932',
    paper_pwc_id='exploring-the-limits-of-weakly-supervised',
    input_transform=b0_input_transform,
    batch_size=256,
    num_gpu=1
)
