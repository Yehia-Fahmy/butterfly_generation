# Butterfly Generation Diffusion Project

This project leverages advanced diffusion models for generating high-quality butterfly images. It includes modules for data management, model architecture, scheduling, data loading, and training. Below is an overview of the key components and their functionality.

## File Structure

### 1. `dataclass.py`
Handles data-related operations:
- Defines the data schema and preprocessing pipeline.
- Manages dataset loading and transformation.
- Supports integration with the `train_dataloader` module.

### 2. `model.py`
Defines the core architecture of the diffusion model:
- Implements layers and building blocks for diffusion operations.
- Optimized for image generation tasks.
- Provides utilities for model saving and loading.

### 3. `scheduler.py`
Manages scheduling for the diffusion process:
- Implements noise scheduling algorithms.
- Controls time steps and denoising strategies.
- Configurable for different diffusion strategies.

### 4. `train_dataloader.py`
Facilitates data loading for training:
- Includes batching, shuffling, and augmentation features.
- Ensures compatibility with the model and training pipeline.
- Optimized for handling large datasets efficiently.

### 5. `train.py`
Orchestrates the training process:
- Integrates the model, dataloader, and scheduler.
- Handles training loops, loss calculations, and optimizations.
- Includes checkpointing and metrics tracking.

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Required Python libraries listed in `requirements.txt`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Yehia-Fahmy/butterfly-generation-diffusion.git
   cd butterfly-generation-diffusion
2. Install dependencies:
    ```bash
    pip install -r requirements.txt

### Usage
1. Modify the dataclass.py file with your desired settings
2. Run the training script:
    ```bash
    python train.py