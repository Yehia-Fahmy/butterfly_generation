import torch
from datasets import load_dataset
from dataclass import TrainingConfig
import matplotlib.pyplot as plt
from torchvision import transforms


class DataLoader():
    """DataLoader class to load the dataset and return the train dataloader"""
    def __init__(self):
        self.config = TrainingConfig()

        self.config.dataset_name = "huggan/smithsonian_butterflies_subset"

        self.dataset = load_dataset(self.config.dataset_name, split="train")
        print("Dataset loaded successfully:")
        print(self.dataset)
       
        def show_sample(self):       
            fig, axs = plt.subplots(1, 4, figsize=(16, 4))
            for i, image in enumerate(self.dataset[:4]["image"]):
                axs[i].imshow(image)
                axs[i].set_axis_off()
            plt.show()

        preprocess = transforms.Compose(
            [
                transforms.Resize((self.config.image_size, self.config.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        def transform(examples):
            images = [preprocess(image.convert("RGB")) for image in examples["image"]]
            return {"images": images}

        self.dataset.set_transform(transform)

    def return_train_dataloader(self):
        """Return the train dataloader"""
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.config.train_batch_size, shuffle=True)

        
data_loader = DataLoader()
train_dataloader = data_loader.return_train_dataloader()
