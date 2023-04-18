

#%% imports
import torch

from PIL import Image
import torchvision.transforms as transforms
import torch
import requests
from io import BytesIO

#%% list of images
urls = {'forest': 'https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg',
        'lake': 'https://www.moillusions.com/wp-content/uploads/2015/08/upside-down-lake.jpg'}
        
def get_img_url(url):
    response = requests.get(url)
    init_img = Image.open(BytesIO(response.content)).convert("RGB")
    init_img = init_img.resize((768, 512))
    return init_img

# Load the image from path
# image_path = "/path/to/image.jpg"
# image = Image.open(image_path)
# image

# get image from url
image = get_img_url(urls['lake'])

# Define the transformation to convert the image to a tensor
transform = transforms.ToTensor()

# Apply the transformation to the image
img = transform(image).unsqueeze(0) # transform to tensor and add batch dimension

image

# %%

print(img.shape)

#%% subsample with grid_sample
import torch.nn.functional as F

d = torch.linspace(-1, 1, 224)
meshx, meshy = torch.meshgrid((d, d))

grid = torch.stack((meshy, meshx), 2)
grid = grid.unsqueeze(0)

warped = F.grid_sample(img, grid, mode='bilinear', align_corners=False)

to_image = transforms.ToPILImage()
im = to_image(img.squeeze())
# plot in matplotlib
import matplotlib.pyplot as plt

plt.imshow(im)
plt.title()
plt.show()

imw = to_image(warped.squeeze()) #.show(title='WARPED')
imw
plt.imshow(imw)
plt.title('Subsampled')
plt.show()
# im
#%% sample a trajectory

N = 1000
d = torch.cumsum(torch.randn(N, 2)*.001, dim=0)
d = d.clamp(-1, 1)

plt.plot(d[:,0].numpy(), d[:,1].numpy(), '-')

grid = d.unsqueeze(0).unsqueeze(0)
grid.shape

coneinputs = F.grid_sample(img, grid, mode='bilinear', align_corners=False)

Rstar = coneinputs.sum(dim=(0,1,2))
plt.plot(Rstar)
# %%
