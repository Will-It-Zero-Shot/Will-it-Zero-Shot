import torch
from PIL import Image
from datasets import load_dataset
import clip
from openai import OpenAI
import numpy as np
import matplotlib.pyplot as plt
import gc
import os
import json
import sys
from scipy.stats import spearmanr
import urllib.request
import time
from azure.storage.blob import BlobServiceClient
import io
import torch
from transformers import CLIPProcessor, CLIPModel
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the dataset
# ds = load_dataset("imageomics/rare-species")
ds = []

# # Inspect dataset splits
# print(ds)

dataset_split = 'train'

# Load CLIP model
# clip_model, clip_preprocess = clip.load("ViT-B/16", device=device)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda()
clip_preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def get_image_embedding(model, preprocessor, image):
    inputs = preprocessor(images=image, return_tensors="pt").to('cuda')
    with torch.no_grad():
        image_embedding = model.get_image_features(**inputs)
    return image_embedding.squeeze()  # Keep on CUDA

def get_text_embedding(model, preprocessor, text):
    inputs = preprocessor(text=text, return_tensors="pt").to('cuda')
    with torch.no_grad():
        text_embedding = model.get_text_features(**inputs)
    return text_embedding.squeeze()

def blob_mongo_save(prompt, buf):
    current_time = round(time.time() * 1000)
    blob_name = f"{prompt.replace(' ', '-')[:100]}-{current_time}"
    connect_str = ''
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    container_name = 'will-clip-zero-shot'
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=f"{blob_name}.png")
    print(blob_client.upload_blob(buf))

    data = {
        "phrase": prompt,
        "image": f"https://.blob.core.windows.net/will-clip-zero-shot/{blob_name}.png",
        "created_at": current_time,
    }
    
    return data
base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
ckpt = "sdxl_lightning_4step_unet.safetensors"

# Load model.
unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("cuda", torch.float16)
unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cuda"))
sdxl_pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16").to("cuda")

# Ensure sampler uses "trailing" timesteps.
sdxl_pipe.scheduler = EulerDiscreteScheduler.from_config(sdxl_pipe.scheduler.config, timestep_spacing="trailing")


def generate_image_sdxl(prompt):
    return sdxl_pipe(prompt=f'a realistic picture of {prompt}').images[0]

def generate_image_dalle(prompt):
    client = OpenAI(api_key="") #

    response = client.images.generate(
    model="dall-e-3",
    prompt=f'a realistic picture of {prompt}',
    size="1024x1024",
    quality="standard",
    response_format="url"
    )

    for gen_image in response.data:
        with urllib.request.urlopen(gen_image.url) as url:
            img = Image.open(url)
            img.convert('RGBA')
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            buf.seek(0)


def will_it_zero_shot(preprocess, item, model, tokenizer, actual_label, label_scores, total_counts, all_labels_full):
    image = preprocess(item['image']).unsqueeze(0).to(device)

    encoded_label = model.encode_text(tokenizer(actual_label).to(device))
    encoded_label /= encoded_label.norm(dim=-1, keepdim=True)

    text = tokenizer(all_labels_full).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)

        text_features = model.encode_text(text)
        logits_per_image = image_features @ text_features.T
        
        # do actual calc
        text_features /= text_features.norm(dim=-1, keepdim=True)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        other_text_embeddings = torch.cat([text_features[:all_labels_full.index(actual_label)], text_features[all_labels_full.index(actual_label)+1:]])
        avg_other_text_embedding = other_text_embeddings.mean(dim=0)

        # Calculate the similarity scores
        sim_ii = (image_features @ encoded_label.T).item()
        sim_ij = (image_features @ avg_other_text_embedding.T).item()

        # Calculate the norm of the difference between the current text embedding and the average of others
        norm_diff = (encoded_label - avg_other_text_embedding).norm().item()

        # Calculate the final score
        score = (sim_ii - sim_ij) / norm_diff
        label_scores[actual_label].append(score)
        total_counts[actual_label] += 1
        return score


def evaluation_batch(batch, batch_num, model, preprocess, tokenizer, device, label_scores, label_top_k_counts, total_counts, all_labels_full, k=5):
    batch_results = []
    device = "cuda"
    for item in batch:
        try:
            with torch.device('cuda'):
                print(item, f"{item['common']}")
                actual_label = f"a picture of {item['common']}".lower()
                score = will_it_zero_shot(preprocess, item, model, tokenizer, actual_label, label_scores, total_counts, all_labels_full)
                
            top_k_indices = probs.argsort()[0][-k:][::-1]
            top_k_labels = [all_labels_full[i] for i in top_k_indices]
            # print(top_k_labels)
            batch_results.append({
                "actual_label": actual_label,
                "actual_label_score": score,
                "top_k_labels": top_k_labels
            })

            if f"a picture of {item['common']}".lower() in top_k_labels:
                label_top_k_counts[actual_label] += 1

            text_features = text = image = image_features = encoded_label = probs = avg_other_text_embedding = other_text_embeddings = None
            del text_features
            del text
            del image
            del image_features
            del encoded_label
            del probs
            del avg_other_text_embedding
            del other_text_embeddings
            del top_k_indices
            del top_k_labels
            torch.cuda.empty_cache()
            gc.collect()

        except SyntaxError as e:
            print('errrr', e)
            return label_scores, label_top_k_counts, total_counts, 0, 0
        
        batch_file_path = os.path.join('batch_scientific', f"batch_{batch_num}_1.json")
        with open(batch_file_path, 'w') as f:
            json.dump(batch_results, f)

        
    return label_scores, label_top_k_counts, total_counts



def save_set_to_file(my_set, filename):
    # Convert set to list of strings
    list_of_strings = list(my_set)
    
    # Write the list to a file
    with open(filename, 'w') as file:
        for item in list_of_strings:
            file.write(f"{item}\n")

def read_file_to_list(filename):
    # Read the file and convert to list
    with open(filename, 'r') as file:
        list_of_strings = [line.strip() for line in file]
    return list_of_strings


def evaluate_model(num, model, preprocess, tokenizer, dataset, device):
    correct = 0
    correct_scientific = 0
    correct_both = 0
    total = 0

    all_labels = set(f"a picture of {item['common']}" for item in dataset)
    save_set_to_file(all_labels, 'all_labels_.txt')
    
    all_labels_full = read_file_to_list('all_labels.txt')


    label_scores = {label: [] for label in all_labels_full}
    label_top_k_counts = {label: 0 for label in all_labels_full}
    total_counts = {label: 0 for label in all_labels_full}

    batch = []
    batch_size = 300
    batch_num = num
    offset = batch_num * batch_size


    for item in dataset.select(list(range(offset, offset + batch_size))):
        batch.append(item)
        if len(batch) == batch_size:
            label_scores, label_top_k_counts, total_counts = evaluation_batch(batch, batch_num, model, preprocess, tokenizer, device, label_scores, label_top_k_counts, total_counts, all_labels_full)
            batch = []
            break
        total += 1
        print(total)

    if len(batch) > 0:
        label_scores, label_top_k_counts, total_counts = evaluation_batch(batch, batch_num, model, preprocess, tokenizer, device, label_scores, label_top_k_counts, total_counts, all_labels_full)

    avg_scores = {label: np.mean(scores) for label, scores in label_scores.items()}
    top_k_ratios = {label: label_top_k_counts[label] / total_counts[label] if total_counts[label] > 0 else 0 for label in all_labels_full}

    accuracy_scientific = correct_scientific / total
    accuracy_both = correct_both / total
    return avg_scores, top_k_ratios, accuracy_scientific, accuracy_both


def plot_results(avg_scores, top_k_ratios, model_name, top_k=5):
    labels = list(avg_scores.keys())
    avg_scores_values = list(avg_scores.values())
    top_k_ratios_values = list(top_k_ratios.values())

    print(len(avg_scores_values), len(top_k_ratios_values))

    correlation, _ = spearmanr(avg_scores_values, top_k_ratios_values)

    plt.figure(figsize=(12, 8))
    plt.scatter(avg_scores_values, top_k_ratios_values, alpha=0.5, label=model_name)
    # plt.xlabel('Average Zero-shot Prediction Score')
    # plt.ylabel(f'Top-{top_k} Accuracy')
    # plt.title(f'Rare species: Zero-shot Prediction Scores vs Accuracy for {model_name} using common name')
    # plt.grid(True)

    plt.annotate(f'Spearman Rank Correlation: {correlation:.2f}', xy=(0.05, 0.95), xycoords='axes fraction', 
             fontsize=15, ha='left', va='top')
    
    for i, label in enumerate(labels):
        plt.annotate('', (avg_scores_values[i], top_k_ratios_values[i]), fontsize=5, alpha=0.7)
    
    # plt.legend()
    plt.xlim((-.05, 0.4))
    plt.show()


def aggregate_results(batch_results_dir='batch_scientific'):
    all_scores = {}
    top_k_counts = {}
    total_counts = {}
    def read_file_to_list(filename):
        # Read the file and convert to list
        with open(filename, 'r') as file:
            list_of_strings = [line.strip() for line in file]
        return list_of_strings
    
    all_labels = read_file_to_list('all_labels_scientific.txt')

    for label in all_labels:
        all_scores[label] = []
        top_k_counts[label] = 0
        total_counts[label] = 0

    for batch_file in os.listdir(batch_results_dir):
        batch_file_path = os.path.join(batch_results_dir, batch_file)
        with open(batch_file_path, 'r') as f:
            batch_results = json.load(f)

        for result in batch_results:
            actual_label = result["actual_label"]
            actual_label_score = result["actual_label_score"]
            top_k_labels = result["top_k_labels"]

            all_scores[actual_label].append(actual_label_score)
            total_counts[actual_label] += 1

            if actual_label in top_k_labels:
                top_k_counts[actual_label] += 1

    avg_scores = {label: np.mean(scores) for label, scores in all_scores.items() if total_counts[label] > 0}
    top_k_ratios = {label: top_k_counts[label] / total_counts[label] if total_counts[label] > 0 else 0 for label in all_labels}
    plot_results(avg_scores, top_k_ratios, 'BioCLIP')
    # with open('CLIP_common_accuracy.json', 'w') as file:
    #                 print('dumping data')
    #                 json.dump(top_k_ratios, file, indent=4)
    # with open('CLIP_common_scores.json', 'w') as file:
    #                 print('dumping data')
    #                 json.dump(avg_scores, file, indent=4)

    return avg_scores, top_k_ratios

err_count = 0

aggregate_results()
# Run Evaluation
# avg_scores, top_k_ratios, clip_accuracy_scientific, clip_accuracy_both = evaluate_model(clip_model, clip_preprocess, clip.tokenize, ds[dataset_split], device)
# def main(num):
#     avg_scores, top_k_ratios, clip_accuracy_scientific, clip_accuracy_both = evaluate_model(num, clip_model, clip_preprocess, clip.tokenize, ds[dataset_split], device)

# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         print("Usage: python script.py <run_number>")
#         sys.exit(1)
    
#     run_number = int(sys.argv[1])
#     main(run_number)
# print(f"CLIP Accuracy: {clip_accuracy_scientific:.2f}, {clip_accuracy_scientific:.2f}, {clip_accuracy_both:.2f}")
# # print(f"BioCLIP Accuracy: {clip_accuracy_scientific:.2f}, {bioclip_accuracy_scientific:.2f}, {bioclip_accuracy_both:.2f}")

# plot_results(avg_scores, top_k_ratios, 'BioCLIP')
# plot_results(avg_scores_bio, top_k_ratios_bio, 'BioCLIP')

