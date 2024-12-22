# Clone the repository and install dependencies
!git clone https://github.com/justinpinkney/stable-diffusion.git
%cd stable-diffusion
!pip install --upgrade pip
!pip install -r requirements.txt

# Upgrade keras and uninstall torchtext if necessary
!pip install --upgrade keras
!pip uninstall -y torchtext

# Install datasets for managing the dataset
!pip install datasets

import os
from PIL import Image
import numpy as np
from datasets import Dataset

# Base path to the project folder
base_path = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/execinstance/code/Users/1155143473/humananimalears"

# Paths to the dataset and captions file
image_dir = os.path.join(base_path, "my_dataset/images")
captions_file = os.path.join(base_path, "my_dataset/captions.txt")

# Step 1: Read Captions File and Prepare Dataset
data = []
with open(captions_file, "r") as f:
    for line in f:
        try:
            # Parse image name and caption from each line
            image_name, caption = map(str.strip, line.strip().split(":", 1))
            image_path = os.path.join(image_dir, image_name)
            if os.path.exists(image_path):
                data.append({"image": image_path, "text": caption})
            else:
                print(f"Image file not found: {image_path}")
        except ValueError:
            print(f"Skipping malformed line: {line.strip()}")

print(f"Loaded {len(data)} samples from local dataset.")

# Step 2: Convert to Hugging Face Dataset if data is available
if data:
    dataset = Dataset.from_dict({
        "image": [item["image"] for item in data],
        "text": [item["text"] for item in data],
    })

# Step 3: Preprocess Images
def preprocess_images(sample):
    """Resize and normalize images."""
    image = Image.open(sample["image"]).convert("RGB")
    image = image.resize((512, 512))  # Resize to 512x512
    sample["image"] = np.array(image) / 255.0  # Normalize to [0, 1]

# Display data for verification
display(data)

# Install ipywidgets for notebook compatibility
!pip install ipywidgets

# Login to Hugging Face Hub
!pip install huggingface_hub
from huggingface_hub import notebook_login
notebook_login()

# Download model checkpoint
from huggingface_hub import hf_hub_download
ckpt_path = hf_hub_download(repo_id="CompVis/stable-diffusion-v-1-4-original", filename="sd-v1-4-full-ema.ckpt", use_auth_token=True)

# Define training parameters
BATCH_SIZE = 3
N_GPUS = 2
ACCUMULATE_BATCHES = 1
gpu_list = ",".join((str(x) for x in range(N_GPUS))) + ","

# Run the training script
!python /mnt/batch/tasks/shared/LS_root/mounts/clusters/execinstance/code/Users/1155143473/humananimalears/stable-diffusion/main.py \
    -t \
    --base configs/mnt/batch/tasks/shared/LS_root/mounts/clusters/execinstance/code/Users/1155143473/humananimalears/stable-diffusion/configs/stable-diffusion/animalfeature.yaml \
    --gpus "0" \
    --scale_lr False \
    --num_nodes 1 \
    --check_val_every_n_epoch 10 \
    --finetune_from "$ckpt_path" \
    data.params.batch_size="$BATCH_SIZE" \
    lightning.trainer.accumulate_grad_batches="$ACCUMULATE_BATCHES" \
    data.params.validation.params.n_gpus="$N_GPUS"

# Generate images using the trained model
!python /mnt/batch/tasks/shared/LS_root/mounts/clusters/execinstance/code/Users/1155143473/humananimalears/stable-diffusion/scripts/txt2img.py \
    --prompt "A young blond woman with big cat ears" \
    --outdir "/mnt/batch/tasks/shared/LS_root/mounts/clusters/execinstance/code/Users/1155143473/humananimalears/outputs" \
    --H 512 --W 512 \
    --n_samples 4 \
    --config "/mnt/batch/tasks/shared/LS_root/mounts/clusters/execinstance/code/Users/1155143473/humananimalears/stable-diffusion/configs/stable-diffusion/animalfeature.yaml" \
    --ckpt "/mnt/batch/tasks/shared/LS_root/mounts/clusters/execinstance/code/Users/1155143473/humananimalears/checkpoints/sd-v2-full-ema.ckpt"
