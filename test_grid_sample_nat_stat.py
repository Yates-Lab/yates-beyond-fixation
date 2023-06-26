

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


plt.imshow(im, extent=[-1, 1, 1, -1])
plt.scatter(meshx.flatten(), meshy.flatten(), s=.1, c='r')
plt.title('Original')
plt.show()

imw = to_image(warped.squeeze()) #.show(title='WARPED')
imw
plt.imshow(imw)
plt.title('Subsampled')
plt.show()
# im
#%% sample a trajectory

N = 1000
d = torch.cumsum(torch.randn(N, 2)*.1, dim=0)
d = d.clamp(-1, 1)

plt.imshow(im, extent=[-1, 1, 1, -1])
plt.title('Original')
plt.plot(d[:,0].numpy(), d[:,1].numpy(), '-r')
plt.show()


grid = d.unsqueeze(0).unsqueeze(0)
grid.shape

coneinputs = F.grid_sample(img, grid, mode='bilinear', align_corners=False)

Rstar = coneinputs.sum(dim=(0,1,2))

plt.plot(Rstar)
plt.show()


# %%

Ngrid = 100
# generate a 2D grid and move it along a trajectory, then use grid_sample to sample from the image
xx,yy = torch.meshgrid(torch.linspace(-1, 1, Ngrid), torch.linspace(-1, 1, Ngrid))

# grid = torch.stack((xx[Ngrid//2,:], yy[Ngrid//2,:]), dim=1)
grid = torch.stack((yy.flatten(), xx.flatten()), dim=1)
grid *= .2

trajectory = torch.cumsum(torch.randn(N, 2)*.01, dim=0)
trajectory = trajectory.clamp(-1, 1)

# concatenate grid and trajectory
grid = trajectory.unsqueeze(0) + grid.unsqueeze(1)
grid = grid.unsqueeze(0)

# sample from image
coneinputs = F.grid_sample(img, grid, mode='bilinear', align_corners=False)

plt.figure(figsize=(10,10))
plt.imshow(coneinputs.mean(dim=(0,1)).T, aspect='auto', interpolation='none')

#%%
import imageio

M = coneinputs.reshape(1, 3, Ngrid, Ngrid, N)
Frames = []

for f in range(N):
    Frames.append(to_image(M[:,:,:, :, f].squeeze()))
    # Frames.append(to_image(M[:,:,:, :, f].squeeze()))
    # plt.imshow(M[:,:,:, :, f].squeeze().T, aspect='auto', interpolation='none')
    # plt.show()
imageio.mimsave('FEMs.gif', Frames, fps=10)

#%%
spacetime = to_image(M[0,:,Ngrid//2,:,:].squeeze().permute(0,2,1))

spacetime


# create a gif from a tensor of images


# %%
I = M[0,:,Ngrid//2,:,:].squeeze().permute(0,2,1)
I = I.mean(dim=0).squeeze()
I = I.log10()
I -= I.mean(dim=0)


sz = I.shape
import numpy as np

window = np.hanning(sz[0])[:,None]*np.hanning(sz[1])[None,:]

# import fft2 from scipy
from scipy.fftpack import fft, fft2, ifft2, fftshift

plt.figure()
plt.imshow(I.numpy(), aspect='auto')


plt.figure()
F = fft2(I.numpy()*window)
Fpow = fftshift(np.log(np.abs(F)))
plt.imshow(Fpow, aspect='auto')

plt.figure()
plt.plot(np.mean(Fpow, axis=0))

F = fft(I[N//2,:].numpy()*window[N//2,:])
Fpow = fftshift(np.log(np.abs(F)))

plt.plot(Fpow)
# %%
