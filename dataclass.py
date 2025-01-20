from dataclasses import dataclass
from huggingface_hub import login
from custom_token import Token

@dataclass
class TrainingConfig:
    def __init__(self):
        # Log in to Hugging Face at object creation
        login(token=Token().secret_token) # Create a token from hugging face with write privilages and paste the string here
        print("Logged in to Hugging Face successfully!")

    image_size = 128  # the generated image resolution
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    # mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    mixed_precision = "no"
    output_dir = "ddpm-butterflies-128"  # the model name locally and on the HF Hub
    dataset_name = ""
    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    overwrite_output_dir = False  # overwrite the old model when re-running the notebook
    seed = 0
