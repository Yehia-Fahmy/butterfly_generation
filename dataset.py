from datasets import load_dataset
from dataclass import TrainingConfig
import matplotlib.pyplot as plt

config = TrainingConfig()

config.dataset_name = "huggan/smithsonian_butterflies_subset"
dataset = load_dataset(config.dataset_name, split="train")

print("Dataset loaded successfully:")
print(dataset)

fig, axs = plt.subplots(1, 4, figsize=(16, 4))
for i, image in enumerate(dataset[:4]["image"]):
    axs[i].imshow(image)
    axs[i].set_axis_off()
plt.show()
