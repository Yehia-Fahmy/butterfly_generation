import torch
from PIL import Image
from diffusers import DDPMScheduler
from dataclass import TrainingConfig
from train_dataloader import DataLoader
from model import Model
import torch.nn.functional as F

config = TrainingConfig()

data_loader = DataLoader()
train_dataloader = data_loader.return_train_dataloader()
model = Model().model

sample_image = data_loader.dataset[0]["images"].unsqueeze(0)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
noise = torch.randn(sample_image.shape)
timesteps = torch.LongTensor([50])
noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)

Image.fromarray(((noisy_image.permute(0, 2, 3, 1) + 1.0) * 127.5).type(torch.uint8).numpy()[0]).show()

noise_pred = model(noisy_image, timesteps).sample
print("noise pred: ")
print(noise_pred)
loss = F.mse_loss(noise_pred, noise)
print("loss")
print(loss)
