#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from torch.optim import AdamW
from accelerate import Accelerator


# In[2]:


# Step 1: Load and preprocess dataset
def preprocess_images(input_path, output_path, size=(512, 512)):
    os.makedirs(output_path, exist_ok=True)
    for file_name in os.listdir(input_path):
        img_path = os.path.join(input_path, file_name)
        if not file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"Skipping non-image file: {file_name}")
            continue
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(size)  # Resize to 512x512 for Stable Diffusion
            img.save(os.path.join(output_path, file_name))
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

# Define paths
input_path = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/execinstance/code/Users/1155143473/Human Image Generation with Animal Features/my_dataset/images"
output_path = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/execinstance/code/Users/1155143473/Human Image Generation with Animal Features/my_dataset/preprocessed_images"
captions_path = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/execinstance/code/Users/1155143473/Human Image Generation with Animal Features/my_dataset/captions.txt"


# Preprocess images
preprocess_images(input_path, output_path)

# Load captions and prepare dataset
data = []
with open(captions_path, "r") as f:
    for line in f:
        image_name, caption = line.strip().split("   ")  # Assuming '   ' separates filename and caption
        image_path = os.path.join(output_path, image_name)
        if os.path.exists(image_path):
            data.append({"image": image_path, "text": caption})

# Create Hugging Face Dataset
from datasets import Dataset
ds = Dataset.from_dict({
    "image": [item["image"] for item in data],
    "text": [item["text"] for item in data],
})

# Preprocess images (resize and normalize)
def preprocess_images(sample):
    image = Image.open(sample["image"]).convert("RGB")
    image = image.resize((512, 512))  # Resize to 512x512 for Stable Diffusion XL
    sample["image"] = np.array(image) / 255.0  # Normalize image to [0, 1]
    return sample

ds = ds.map(preprocess_images, batched=False)


# In[3]:


# Step 2: Load model components
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
device = "cuda" if torch.cuda.is_available() else "cpu"

unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")


# In[4]:


# Step 3: Create DataLoader with correct collate function
def collate_fn(batch):
    images = torch.tensor([item["image"] for item in batch]).permute(0, 3, 1, 2).float()  # Convert to tensor and reorder to (B, C, H, W)
    captions = [item["text"] for item in batch]
    return {"images": images.to(device), "captions": captions}

dataloader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_fn)


# In[6]:


# Step 4: Set up optimizer and Accelerator
optimizer = AdamW(unet.parameters(), lr=5e-6)
accelerator = Accelerator(mixed_precision="fp16")
unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)


# In[20]:


# Step 5: Simplified Training Loop

for epoch in range(3):  # Training for 3 epochs
    unet.train()  # Set the UNet model to training mode
    for step, batch in enumerate(dataloader):
        # Get the images and captions from the batch
        images = batch["images"].to(device)
        captions = batch["captions"]

        # Tokenize the captions (Ensure they're properly padded and truncated)
        inputs = tokenizer(captions, padding="max_length", truncation=True, max_length=77, return_tensors="pt").to(device)

        # Get the encoder hidden states for the text inputs (captions)
        encoder_hidden_states = text_encoder(inputs.input_ids)[0]

        # Generate random noise for the diffusion process (the noise for each image)
        noise = torch.randn_like(images).to(device)

        # Create timesteps for the diffusion process (uniformly sampled from available timesteps)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (images.shape[0],), device=device).long()

        # Add the noise to the images at these timesteps using the noise scheduler
        noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

        # Ensure `timesteps` has the correct shape: [batch_size, seq_len, 1]
        timesteps = timesteps.unsqueeze(1)  # Shape: [batch_size, 1]
        timesteps = timesteps.expand(-1, encoder_hidden_states.shape[1])  # Shape: [batch_size, seq_len]
        timesteps = timesteps.unsqueeze(-1)  # Shape: [batch_size, seq_len, 1] to match encoder_hidden_states

        # Forward pass through the UNet model: Pass in noisy images, timesteps, and encoder hidden states
        noise_pred = unet(
            noisy_images, 
            timesteps, 
            encoder_hidden_states, 
            added_cond_kwargs={
                "text_embeds": encoder_hidden_states,  # The text embeddings
                "time_ids": timesteps  # The time embeddings corresponding to diffusion steps
            }
        ).sample()

        # Compute the loss between the predicted noise and the actual noise added to the images
        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        # Perform backward pass and optimizer step
        accelerator.backward(loss)

        if (step + 1) % 2 == 0:  # Gradient accumulation every 2 steps
            optimizer.step()
            optimizer.zero_grad()

        # Log progress (you can adjust the frequency as needed)
        if step % 10 == 0:
            print(f"Epoch {epoch + 1}/{3}, Step {step + 1}, Loss: {loss.item()}")


# In[ ]:


# Step 6: Save fine-tuned model
unet.save_pretrained("/mnt/batch/tasks/shared/LS_root/mounts/clusters/execinstance/code/Users/1155143473/Human Image Generation with Animal Features/model/stable-diffusion-xl-finetuned/unet")
text_encoder.save_pretrained("/mnt/batch/tasks/shared/LS_root/mounts/clusters/execinstance/code/Users/1155143473/Human Image Generation with Animal Features/model/stable-diffusion-xl-finetuned/text_encoder")

