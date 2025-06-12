import torch
import pandas as pd
import os
import csv
from datasets import load_dataset
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from azure.storage.blob import BlobServiceClient
from transformers import CLIPProcessor, CLIPModel, AutoModel, AutoProcessor
from transformers import FlavaProcessor, FlavaModel, FlavaFeatureExtractor, BertTokenizer
from pymongo import MongoClient
import pickle
import io
import torch
import openai
import urllib.request
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from PIL import Image
import time
import threading
import json
import numpy as np
import argparse
import importlib
from open_clip import create_model_from_pretrained, get_tokenizer
import concurrent.futures
from sklearn.metrics import silhouette_samples
from cifar_categories import cifar_category_map
from collections import defaultdict
from glob import glob

# Main variables
category_in_prompt = ''
# dataset_name = 'aircraft'
# generation_tool = 'sdxl'
# model_name = 'clip'
# USE_GENERATED = True
# dataset_link = "clip-benchmark/wds_fgvc_aircraft"

# CHANGE WHEN SWITCHING DATASETS
# from aircraft_classnames import label_to_class

def main(dataset_name, dataset_link, generation_tool, model_name, use_generated, image_size, local=False, output_dir=None):
    print(f"Running with parameters: dataset_name, generation_tool, model_name")
    module_name = "math"  # This can be set dynamically

    class_names_lib = importlib.import_module(f"{dataset_name if dataset_name != 'vegfru' else dataset_link}_classnames")
    device = "cuda"
    label_to_class = class_names_lib.label_to_class

    # Load the dataset
    dataset_split = 'test' if dataset_name in ['food', 'sun', 'objectnet', 'imagenet'] else 'train'
    # if dataset_name in ['objectnet', 'imagenet', 'sun']:
    #     dataset_split = dataset_split + '[:10%]'
    if not local:
        ds = load_dataset(dataset_link, split=dataset_split) if dataset_name != 'flowers' else load_dataset("webdataset", data_files={"test": "https://huggingface.co/datasets/clip-benchmark/wds_vtab-flowers/resolve/main/test/*.tar"}, split="test")
    else:
        rows = []
        test_file = None
        for filename in ["sneaker_dataset.csv", "gemstone_dataset.csv", "butterflies and moths.csv", "vegfru_list\\vegfru_test.txt"]:
            file_path = os.path.join('.', dataset_name, filename)
            print(file_path)
            if os.path.isfile(file_path):
                test_file = file_path
                break

        if test_file is None:
            raise FileNotFoundError("No test.csv or test.txt file found in the provided local dataset directory.")

        if test_file.endswith(".csv"):
            with open(test_file, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if dataset_name not in ['gemstones', 'sneakers'] and row.get("data set") != dataset_split:
                        continue
                    try:
                        img_path = os.path.join(dataset_name, *row["filepaths"].split('/'))
                        
                        rows.append({"cls": row["class id"], "webp": img_path})
                    except Exception as e:
                        print(f"Failed to load image {img_path}: {e}")
        else:
            with open(test_file, encoding='utf-8') as txtfile:
                for line in txtfile:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        path_part, cls_part = line.split(" ", 1)
                        img_path = os.path.join(dataset_name, path_part)
                        
                        rows.append({"cls": cls_part.strip(), "webp": img_path})
                    except Exception as e:
                        print(f"Failed to load image from line '{line}': {e}")
        ds = rows

    # print(ds)

    if model_name == 'siglip':
        model = AutoModel.from_pretrained("google/siglip-base-patch16-512").cuda()
        image_processor = text_processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-512")
    elif model_name == 'flava':
        model = FlavaModel.from_pretrained("facebook/flava-full").cuda()
        image_processor = FlavaFeatureExtractor.from_pretrained("facebook/flava-full")
        text_processor = BertTokenizer.from_pretrained("facebook/flava-full")
    else:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda()
        image_processor = text_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    image_embeddings = {}
    text_embeddings = {}

    mongo_connection_string = ""
    db_client = MongoClient(mongo_connection_string)
    db_name = "data_experimental"
    db = db_client[db_name]
    collection = db["generated_datasets"]

    def get_image_embedding(model, preprocessor, image):
        try:
            inputs = preprocessor(images=image.convert("RGB"), return_tensors="pt").to('cuda')
            with torch.no_grad():
                image_embedding = model.get_image_features(**inputs)
                if model_name in ['flava']:
                    image_embedding = image_embedding[:, 0].float()
            image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
            return image_embedding.squeeze()  # Keep on CUDA
        except Exception as ex:
            print(ex)
            return None

    def get_text_embedding(model, preprocessor, text):
        try:
            inputs = preprocessor(text=text, return_tensors="pt", padding="max_length").to('cuda')
            with torch.no_grad():
                text_embedding = model.get_text_features(**inputs)
                if model_name in ['flava']:
                    text_embedding = text_embedding[:, 0].float()
            text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
            return text_embedding.squeeze()
        except Exception as ex:
            print(ex)
            return None
        
    def embed_row(
        row, 
        model, 
        image_processor, 
        label_to_class, 
        class_tag, 
        img_tag, 
        category_in_prompt
    ):
        """
        Extract the class_name from the row, compute the embedding,
        and return (class_name, embedding).
        """
        class_id = row[class_tag]
        # Decide how we look up the class name
        if category_in_prompt == 'nested':
            class_name = label_to_class[class_id]
        else:
            class_name = list(label_to_class.values())[class_id]

        # Compute embedding
        print(row)
        embedding = get_image_embedding(model, image_processor, row[img_tag])
        return (class_name, embedding)

    def generate_embeddings_multithreaded(
        ds,
        dataset_split,
        label_to_class,
        class_tag,
        img_tag,
        model,
        image_processor,
        category_in_prompt,
        max_workers=10
    ):
        """
        Reads each row in ds[dataset_split], gets its class_name, computes its embedding,
        and stores it in image_embeddings[class_name].
        """
        image_embeddings = defaultdict(list)

        # Convert dataset (ds[dataset_split]) to a list or iterable 
        # rows = list(ds)

        # Use ThreadPoolExecutor to process rows concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            # 1. Schedule each row as a task
            for row in rows:
                futures.append(
                    executor.submit(
                        embed_row,
                        row,
                        model,
                        image_processor,
                        label_to_class,
                        class_tag,
                        img_tag,
                        category_in_prompt
                    )
                )

            # 2. As tasks complete, add the results to the correct class_name’s list
            for future in concurrent.futures.as_completed(futures):
                class_name, embedding = future.result()
                if embedding is not None:
                    image_embeddings[class_name].append(embedding)

        return image_embeddings
        

    class_tag = 'cls'
    img_tag = 'webp'
    if dataset_name in ['aircraft', 'cars', 'imagenet']:
        img_tag = 'jpg'
    elif dataset_name == 'cub' or dataset_name == 'textures':
        img_tag = 'image'
        class_tag = 'label'
    elif dataset_name == 'objectnet':
        img_tag = 'png'
    
    # print(ds.features[class_tag], )
    num = 0 
    for class_id, class_name in enumerate(label_to_class.keys() if category_in_prompt == 'nested' else label_to_class.values()):
        dataset_prompts = {
            "aircraft": f'a photo of a {class_name.replace("_", " ")}, a type of aircraft.',
            "resisc": f'a sattelite image of a {class_name.replace("_", " ")}',
            "cub": f'a photo of a {class_name.replace("_", " ")}, a type of bird.',
            "cifar": f'a photo of a {class_name.replace("_", " ")}, a type of {cifar_category_map[class_name.replace("_", " ")] if dataset_name == "cifar" else ""}.'
        }
        # print(dataset_prompts.get(dataset_name))
        # text_embeddings[class_name] = get_text_embedding(model, text_processor, dataset_prompts.get(dataset_name))
        text_for_embedding = f'a photo of a {class_name.replace("_", " ")}'
        if output_dir == 'Flamingo':
            text_for_embedding = f'{class_name.replace("_", " ")}'
        elif output_dir == 'photoOfBirdFlamingo':
            text_for_embedding = dataset_prompts.get(dataset_name)
        text_embedding = get_text_embedding(model, text_processor, text_for_embedding)
        text_embeddings[class_name] = text_embedding.detach().cpu()
        image_embeddings[class_name] = []

    if not use_generated: 
        # image_embeddings = generate_embeddings_multithreaded(
        #     ds,
        #     dataset_split,
        #     label_to_class,
        #     class_tag,
        #     img_tag,
        #     model,
        #     image_processor,
        #     category_in_prompt
        # )
        for row in ds:
            class_id = row[class_tag]
            if (dataset_link == 'veg' and int(class_id) > 199) or (dataset_link == 'fru' and int(class_id) <= 199):
                continue
            if dataset_link == 'fru':
                    class_id = int(class_id) - 200
            class_name = label_to_class[int(class_id)] if category_in_prompt == 'nested' else list(label_to_class.values())[int(class_id)]
            image = row[img_tag]
            if isinstance(image, str):
                image = Image.open(row[img_tag]).convert("RGB")
            image_embedding = get_image_embedding(model, image_processor, image)
            print('embedded', row)
            if image_embedding != None:
                cpu_embedding = image_embedding.detach().cpu()
                image_embeddings[class_name].append(cpu_embedding)
            image.close()
            del image_embedding, image, row
            torch.cuda.empty_cache()
            # collection.insert_one({
            #         'dataset': dataset_name,
            #         'generation_tool': 'real',
            #         'class_name': class_name.replace(" ", "-")[:100],
            #         'model': model_name,
            #         'num': None,
            #         'url': row['__url__'],
            #         'embedding': pickle.dumps(image_embedding),
            #     })

        class_means = {class_name: torch.mean(torch.stack(image_embeddings[class_name]), dim=0) for class_name in image_embeddings}

    generated_image_embeddings = {}

    mongo_connection_string = ""
    db_client = MongoClient(mongo_connection_string)
    db_name = "data_experimental"
    db = db_client[db_name]
    collection = db["generated_datasets"]

    def process_image(blob_url, model, image_processor, image_size):
        try:
            with urllib.request.urlopen(blob_url) as url:
                print(blob_url)
                with Image.open(url).convert("RGB") as img:
                    img = img.resize((image_size, image_size))
                    embedding = get_image_embedding(model, image_processor, img)
                    return embedding
        except Exception as e:
            print(f"Failed to process {blob_url} due to: {e}")
            return None

    if use_generated:
        # 1. Build the flat list of (class_name, blob_url)
        tasks = []
        prefix = 'captions' if generation_tool == 'sdxl' else 'photo'
        for class_name in label_to_class.values():
            generated_image_embeddings[class_name] = []
            for i in range(20):
                blob_name = f'{prefix}-{dataset_name}-{class_name.replace(" ", "-")[:100]}-{generation_tool}-{i}'
                blob_url  = f"https://.blob.core.windows.net/will-clip-zero-shot/{blob_name}.png"
                tasks.append((class_name, blob_url))

        # 2. Helper to process a single (class_name, url) pair
        def fetch_and_embed(pair):
            class_name, url = pair
            emb = process_image(url, model, image_processor, image_size)
            if emb is None:
                return None
            # free GPU immediately
            cpu_emb = emb.detach().cpu()
            del emb
            return class_name, cpu_emb

        # 3. Execute in batches
        batch_size = 200
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            for start in range(0, len(tasks), batch_size):
                batch = tasks[start:start + batch_size]
                futures = [executor.submit(fetch_and_embed, p) for p in batch]

                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result is not None:
                        class_name, cpu_emb = result
                        generated_image_embeddings[class_name].append(cpu_emb)
                # cleanup
                for f in futures:
                    del f
                torch.cuda.empty_cache()

        # 4. Optional sanity check
        for class_name, embs in generated_image_embeddings.items():
            print(f"{class_name}: {len(embs)} embeddings")      
            class_means = {class_name: torch.mean(torch.stack(generated_image_embeddings[class_name]), dim=0) for class_name in generated_image_embeddings}



    def calculate_zero_shot_accuracy(class_name, embeddings):
        correct_predictions = 0
        total_predictions = len(embeddings[class_name])

        for I_k in embeddings[class_name]:
            # Calculate cosine similarities between image embedding and each class's text embedding
            print(I_k.unsqueeze(0).shape, text_embeddings[class_name].unsqueeze(0).shape)
            similarities = [torch.nn.functional.cosine_similarity(I_k.unsqueeze(0), text_embeddings[cname].unsqueeze(0)).item()
                            for cname in text_embeddings]
            # Find the class with the highest similarity
            predicted_class_index = similarities.index(max(similarities))
            predicted_class = list(text_embeddings.keys())[predicted_class_index]
            print(predicted_class, class_name)
            # Check if the prediction is correct
            if predicted_class == class_name:
                correct_predictions += 1
        print(class_name, correct_predictions / total_predictions)
        return correct_predictions / total_predictions

    # def calculate_consistency_score(
    #     image_embeddings,
    #     class_means,
    #     text_embeddings,
    #     target_class,
    #     alpha=10.0
    # ):
    #     """
    #     Calculates a "consistency" score for a given target_class by 
    #     using soft exponential weights instead of a hard threshold.

    #     Parameters
    #     ----------
    #     image_embeddings : dict of {class_name: list of np.array}
    #         Each key is a class label, each value is a list of image embeddings for that class.
    #     class_means : dict of {class_name: torch.Tensor}
    #         The centroid (mean) for each class in image embedding space.
    #     text_embeddings : dict of {class_name: torch.Tensor}
    #         The text embedding for each class label.
    #     target_class : str
    #         The class we are computing the score for.
    #     alpha : float
    #         Controls how sharply we weight classes by text-text similarity:
    #         w_j = exp(alpha * cos(T_target, T_j)).
    #         Larger alpha => more emphasis on the most similar classes.

    #     Returns
    #     -------
    #     S_i : float
    #         The final consistency score for target_class (a weighted average
    #         of directional similarities).
    #     """

    #     # 1) Collect the image embeddings for the target_class
    #     target_class_images = torch.stack(
    #         [torch.tensor(img, dtype=torch.float) for img in image_embeddings[target_class]]
    #     )
    #     T_target = text_embeddings[target_class]

    #     # 2) Precompute weights for each class j != target_class based on text-text similarity
    #     #    w_j = exp(alpha * cos(T_target, T_j)).
    #     weights = {}
    #     for class_j, T_j in text_embeddings.items():
    #         if class_j == target_class:
    #             continue
    #         # text-text similarity
    #         sim_j = torch.nn.functional.cosine_similarity(T_target.unsqueeze(0), T_j.unsqueeze(0)).item()
    #         w_j = torch.exp(torch.tensor(alpha * sim_j)).item()  # scalar
    #         weights[class_j] = w_j

    #     # If all weights end up being extremely small or zero, handle that (e.g. if alpha is huge)
    #     sum_w = sum(weights.values())
    #     if sum_w < 1e-8:
    #         # Fallback: no meaningful neighbor? Just return 0
    #         return 0.0

    #     # 3) For each image I_k in the target class, compute a weighted average of
    #     #    directional similarities to each class j, with weight = w_j
    #     #    directional similarity = cos((I_k - mean_j), (T_target - T_j)).
    #     s_k_scores = []
    #     for I_k in target_class_images:
    #         # We'll accumulate sum_of( w_j * direction_sim ) over j, then divide by sum_w
    #         sum_weighted_sim = 0.0

    #         for class_j, w_j in weights.items():
    #             # Directional similarity
    #             diff_image_vector = I_k - class_means[class_j]
    #             diff_text_vector  = T_target - text_embeddings[class_j]

    #             direction_sim = torch.nn.functional.cosine_similarity(
    #                 diff_image_vector.unsqueeze(0),
    #                 diff_text_vector.unsqueeze(0),
    #                 dim=1
    #             ).item()

    #             # Accumulate weighted
    #             sum_weighted_sim += (w_j * direction_sim)

    #         # Weighted average for this image
    #         avg_sim_for_image = sum_weighted_sim / sum_w
    #         s_k_scores.append(avg_sim_for_image)

    #     # 4) Average across all images to get final S_i
    #     if s_k_scores:
    #         S_i = torch.tensor(s_k_scores).mean().item()
    #     else:
    #         S_i = 0.0  # If no images or something degenerate

    #     return S_i

    def calculate_consistency_score(
        image_embeddings,   # dict: class -> [image_emb, ...]
        class_means,        # dict: class -> centroid tensor
        text_embeddings,    # dict: class -> text_emb tensor
        target_class: str,
        k: int = 100,
    ):
        # (1) Collect images for target_class
        I_list = image_embeddings[target_class]
        T_target = text_embeddings[target_class]

        # (2) Find top-k neighbors by text similarity
        #     sim_j = cos(T_target, T_j)
        sims = []
        for cls_j, T_j in text_embeddings.items():
            if cls_j == target_class:
                continue
            sim_j = torch.nn.functional.cosine_similarity(
                T_target.unsqueeze(0), T_j.unsqueeze(0), dim=1
            ).item()
            sim_j_img = torch.nn.functional.cosine_similarity(
                class_means[target_class].unsqueeze(0), class_means[cls_j].unsqueeze(0), dim=1
            ).item()
            sims.append((cls_j, sim_j + sim_j_img))
        # sort descending, pick top-k
        sims.sort(key=lambda x: x[1], reverse=True)
        top_k_list = sims[:k]
        print('variance:', top_k_list[0][1] - top_k_list[-1][1])
        variance_sims = top_k_list[0][1] - top_k_list[-1][1]

        # (4) For each image in target_class, compute weighted average directional similarity
        s_k_scores = []
        for img_emb in I_list:
            I_k = torch.tensor(img_emb, dtype=torch.float)
            dir_sims = []
            for (cls_j, _) in top_k_list:
                diff_img = I_k - class_means[cls_j]
                diff_txt = T_target - text_embeddings[cls_j]
                dir_sim = torch.nn.functional.cosine_similarity(
                    diff_img.unsqueeze(0), diff_txt.unsqueeze(0), dim=1
                ).item()
                dir_sims.append((dir_sim, cls_j))
            sorted_similarities = sorted(dir_sims, key=lambda x: x[0])
            s_k_scores.append(sorted_similarities[0][0])
        if s_k_scores:
            S_i = float(torch.tensor(s_k_scores).mean().item())
        else:
            S_i = 0.0

        return S_i
    
    def calculate_consistency_score_2(
        image_embeddings,   # dict: class -> [image_emb, ...]
        class_means,        # dict: class -> centroid tensor
        text_embeddings,    # dict: class -> text_emb tensor
        target_class: str,
        k: int = 100,
    ):
        # (1) Collect images for target_class
        I_list = image_embeddings[target_class]
        T_target = text_embeddings[target_class]

        # (2) Find top-k neighbors by text similarity
        #     sim_j = cos(T_target, T_j)
        sims = []
        for cls_j, T_j in text_embeddings.items():
            if cls_j == target_class:
                continue
            sim_j = torch.nn.functional.cosine_similarity(
                T_target.unsqueeze(0), T_j.unsqueeze(0), dim=1
            ).item()
            sim_j_img = torch.nn.functional.cosine_similarity(
                class_means[target_class].unsqueeze(0), class_means[cls_j].unsqueeze(0), dim=1
            ).item()
            sims.append((cls_j, sim_j + sim_j_img))
        # sort descending, pick top-k
        sims.sort(key=lambda x: x[1], reverse=True)
        top_k_list = sims[:k]
        print('variance:', top_k_list[0][1] - top_k_list[-1][1])
        variance_sims = top_k_list[0][1] - top_k_list[-1][1]
        alpha = 10
        C_i_image = torch.mean(torch.tensor([
            1 - torch.nn.functional.cosine_similarity(I_k.unsqueeze(0), class_means[target_class].unsqueeze(0)).item()
            for I_k in I_list
        ])).item()
        C_j_list = []
        for class_j, sim_j in top_k_list:
            C_j_image = torch.mean(torch.tensor([
                1 - torch.nn.functional.cosine_similarity(I_k.unsqueeze(0), class_means[class_j].unsqueeze(0)).item()
                for I_k in image_embeddings[class_j]
            ])).item()
            C_j_list.append(C_j_image)
        C_j_image_avg = torch.mean(torch.tensor(C_j_list)).item()
        # (3) compute weights from these top-k
        weights = {}
        for cls_j, sim_val in top_k_list:
            w_j = torch.exp(torch.tensor(alpha * sim_val)).item()
            weights[cls_j] = w_j

        sum_w = sum(weights.values())

        # (4) For each image in target_class, compute weighted average directional similarity
        s_k_scores = []
        for img_emb in I_list:
            I_k = torch.tensor(img_emb, dtype=torch.float)
            sum_weighted_sim = 0.0
            for (cls_j, _) in top_k_list:
                w_j = weights[cls_j]
                diff_img = I_k - class_means[cls_j]
                diff_txt = T_target - text_embeddings[cls_j]
                dir_sim = torch.nn.functional.cosine_similarity(
                    diff_img.unsqueeze(0), diff_txt.unsqueeze(0), dim=1
                ).item()
                sum_weighted_sim += w_j * dir_sim

            if sum_w > 1e-8:
                s_k_scores.append(sum_weighted_sim / sum_w)
            else:
                s_k_scores.append(0.0)

        if s_k_scores:
            S_i = float(torch.tensor(s_k_scores).mean().item())
        else:
            S_i = 0.0

        return S_i, C_j_image_avg / 1



    def compute_weights(T_i, T_all, alpha=1.0):
        # 1) Compute cosine similarities with all classes (excluding i if desired)
        similarities = {}
        for class_j, T_j in T_all.items():
            sim = torch.nn.functional.cosine_similarity(
                    T_i.unsqueeze(0),
                    T_j.unsqueeze(0)
                ).item()
            if sim < 1:
                similarities[class_j] = sim

        # 2) Sort by similarity descending, pick top-K
        #    (If T_all includes the same class i, we skip it or drop it)
        sorted_by_sim = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        top_k = sorted_by_sim

        # 3) Softmax over those top-K
        #    w(i,j) = exp(alpha * cos(T_i, T_j)) / sum( ... )
        exps = [np.exp(alpha * sim) for (_, sim) in top_k]
        denom = sum(exps)

        weights = {}
        for idx, (class_j, sim) in enumerate(top_k):
            w_ij = exps[idx] / denom
            weights[class_j] = w_ij
        return weights

    # def calculate_consistency_score(image_embeddings, class_means, text_embeddings, target_class):
    #     s_k_scores = []
    #     target_text_embedding = text_embeddings[target_class]
    #     weights = compute_weights(target_text_embedding, text_embeddings)
    #     # Step 1: Calculate cosine similarity between target_class text embedding and all other class text embeddings
    #     class_similarities = []
    #     for j, text_embedding_j in text_embeddings.items():
    #         if j != target_class:
    #             similarity = torch.nn.functional.cosine_similarity(
    #                 target_text_embedding.unsqueeze(0),
    #                 text_embedding_j.unsqueeze(0)
    #             ).item() + torch.nn.functional.cosine_similarity(
    #                 class_means[target_class].unsqueeze(0),
    #                 class_means[j].unsqueeze(0)
    #             ).item()
    #             class_similarities.append((similarity, j))

    #     # Sort classes by similarity and pick the 3 closest
    #     threshold = 1.6
    #     closest_classes = [item[1] for item in class_similarities if item[0] >= threshold]
    #     # if need smaller go down
    #     while len(closest_classes) < 3:
    #         threshold -= 0.05
    #         closest_classes = [item[1] for item in class_similarities if item[0] >= threshold]

    #     # Step 2: Perform calculations only for the closest classes
    #     target_class_images = torch.stack([torch.tensor(img) for img in image_embeddings[target_class]])
    #     for I_k in target_class_images:
    #         similarities = []
    #         for j in closest_classes:
    #             mean_j = class_means[j]
    #             diff_image_vector = I_k - mean_j
    #             diff_text_vector = target_text_embedding - text_embeddings[j]

    #             # Calculate cosine similarity between I_k and mean_i for weighting
    #             img_quality_weight = torch.nn.functional.cosine_similarity(
    #                 I_k.unsqueeze(0),
    #                 mean_j.unsqueeze(0)
    #             ).item()

    #             # Calculate similarity and apply weight
    #             similarity = torch.nn.functional.cosine_similarity(
    #                 diff_image_vector.unsqueeze(0),
    #                 diff_text_vector.unsqueeze(0)
    #             ).item()
    #             modality_gap = torch.nn.functional.cosine_similarity(
    #                 class_means[target_class].unsqueeze(0),
    #                 target_text_embedding.unsqueeze(0)
    #             ).item()
    #             print('found', similarity * weights[j], img_quality_weight)
    #             weighted_similarity = (similarity * weights[j]) * (img_quality_weight - modality_gap)

    #             similarities.append((weighted_similarity, j))

    #         # Sort similarities and get the lowest 3
    #         sorted_similarities = sorted(similarities, key=lambda x: x[0])
    #         lowest_3_similarities = sorted_similarities
    #         print(f'For class {target_class}, lowest 3 weighted similarities with diffDist penalty: {lowest_3_similarities}')

    #         # Append the smallest weighted similarity value to s_k_scores
    #         s_k_scores.append(lowest_3_similarities[0][0])

    #     # Return the average consistency score S_i for the class
    #     S_i = torch.tensor(s_k_scores).mean().item()
    #     return S_i

    def get_silhouettes(image_embeddings):
        data = []
        labels = []

        for class_label, embeddings in image_embeddings.items():
            cluster_points = [torch.tensor(img).cpu().numpy() for img in embeddings]
            data.extend(cluster_points)
            labels.extend([class_label] * len(cluster_points))

        # Convert to numpy arrays for sklearn silhouette calculation
        data = np.array(data)
        labels = np.array(labels)

        # Compute silhouette scores for all images
        silhouette_scores = silhouette_samples(data, labels)
        return labels,silhouette_scores

    # def calculate_consistency_score_2(image_embeddings, class_means, text_embeddings, target_class, labels, silhouette_scores):
    #     s_k_scores = []
    #     target_text_embedding = text_embeddings[target_class]
    #     target_class_images = torch.stack([torch.tensor(img) for img in image_embeddings[target_class]])

    #     # Map silhouette scores to each image in the target class
    #     target_silhouette_scores = silhouette_scores[labels == target_class]
    #     # Calculate consistency score
    #     for idx, I_k in enumerate(target_class_images):
    #         silhouette_score = target_silhouette_scores[idx]  # Silhouette score for the current image
    #         similarities = []

    #         for j, mean_j in class_means.items():
    #             if j != target_class:
    #                 diff_image_vector = I_k - mean_j
    #                 diff_text_vector = target_text_embedding - text_embeddings[j]

    #                 similarity = torch.nn.functional.cosine_similarity(
    #                     diff_image_vector.unsqueeze(0),
    #                     diff_text_vector.unsqueeze(0)
    #                 ).item()

    #                 # Weight similarity by silhouette score
    #                 weighted_similarity = similarity * (1 + silhouette_score)
    #                 similarities.append((weighted_similarity, j))

    #         # Append the smallest weighted similarity value to s_k_scores
    #         top_sims = [sim for sim, _ in similarities]
    #         s_k_scores.append(sum(top_sims)/len(top_sims))

    #     # Return the average consistency score S_i for the class
    #     S_i = torch.tensor(s_k_scores).mean().item()
    #     return S_i, np.mean(target_silhouette_scores)


    def torch_cosine_distance(u: torch.Tensor, v: torch.Tensor) -> float:
        # Cosine similarity in PyTorch for 1D vectors:
        cos_sim = torch.nn.functional.cosine_similarity(u.unsqueeze(0), v.unsqueeze(0), dim=1)
        # cos_sim is a 1-element tensor
        return float(1.0 - cos_sim.item())

    def calculate_consistency_score_3(
        image_embeddings, class_means, text_embeddings, target_class, lambda_=2.5, top_k=100
    ):
        s_k_scores = []
        target_text_embedding = text_embeddings[target_class]
        target_class_images = torch.stack([torch.tensor(img) for img in image_embeddings[target_class]])

        # Rank classes by similarity and select the top k most similar classes
        class_similarities = []
        for class_j, text_embedding_j in text_embeddings.items():
            if class_j != target_class:
                class_similarity = torch.nn.functional.cosine_similarity(
                    target_text_embedding.unsqueeze(0),
                    text_embedding_j.unsqueeze(0)
                ).item()
                class_similarities.append((class_similarity, class_j))

        # Sort classes by similarity and select the top k
        sorted_classes = sorted(class_similarities, key=lambda x: x[0], reverse=True)[:top_k]
        valid_classes = [class_j for _, class_j in sorted_classes]

        for I_k in target_class_images:
            # -------------------------
            # Compute a(I_k)
            # distance from I_k to its own class centroid + lambda * distance to its text embedding
            dist_Ik_mean_i = torch_cosine_distance(I_k, class_means[target_class])
            dist_Ik_text_i = torch_cosine_distance(I_k, target_text_embedding)
            a_Ik = dist_Ik_mean_i + lambda_ * dist_Ik_text_i

            # -------------------------
            # Compute b(I_k) and aggregate alignment similarities
            inter_dists = []
            for class_j in valid_classes:
                mean_j = class_means[class_j]
                text_j = text_embeddings[class_j]

                # Compute distances for b(I_k)
                dist_Ik_mean_j = torch_cosine_distance(I_k, mean_j)
                dist_Ik_text_j = torch_cosine_distance(I_k, text_j)
                total_dist_j = dist_Ik_mean_j + lambda_ * dist_Ik_text_j
                inter_dists.append(total_dist_j)

            if len(inter_dists) > 0:
                b_Ik = min(inter_dists)
            else:
                # If there are no valid classes
                b_Ik = 0.0

            # -------------------------
            # Multimodal silhouette for this image
            denom = max(a_Ik, b_Ik)
            if denom < 1e-8:
                sil_value = 0.0
            else:
                sil_value = (b_Ik - a_Ik) / denom
            s_k_scores.append(sil_value)

        # Compute final scores
        S_sil_i = torch.tensor(s_k_scores).mean().item() if s_k_scores else 0.0
        return S_sil_i



    
    # def calculate_consistency_score_2(image_embeddings, class_means, text_embeddings, target_class):
    #     s_k_scores = []
    #     target_text_embedding = text_embeddings[target_class]
    #     weights = compute_weights(target_text_embedding, text_embeddings)
    #     # Step 1: Calculate cosine similarity between target_class text embedding and all other class text embeddings
    #     class_similarities = []
    #     for j, text_embedding_j in text_embeddings.items():
    #         if j != target_class:
    #             similarity = torch.nn.functional.cosine_similarity(
    #                 target_text_embedding.unsqueeze(0),
    #                 text_embedding_j.unsqueeze(0)
    #             ).item() + torch.nn.functional.cosine_similarity(
    #                 class_means[target_class].unsqueeze(0),
    #                 class_means[j].unsqueeze(0)
    #             ).item()
    #             class_similarities.append((similarity, j))

    #     # Sort classes by similarity and pick the 5 closest
    #     threshold = 1.6
    #     closest_classes = [item[1] for item in class_similarities if item[0] >= threshold]
    #     # if need smaller go down
    #     while len(closest_classes) < 3:
    #         threshold -= 0.05
    #         closest_classes = [item[1] for item in class_similarities if item[0] >= threshold]
    #     # Step 2: Perform calculations only for the closest classes
    #     target_class_images = torch.stack([torch.tensor(img) for img in image_embeddings[target_class]])
    #     C_i_image = torch.mean(torch.tensor([
    #         1 - torch.nn.functional.cosine_similarity(I_k.unsqueeze(0), class_means[target_class].unsqueeze(0)).item()
    #         for I_k in target_class_images
    #     ])).item()
    #     for I_k in target_class_images:
    #         similarities = []
    #         for j in closest_classes:
    #             mean_j = class_means[j]
    #             diff_image_vector = I_k - mean_j
    #             diff_text_vector = target_text_embedding - text_embeddings[j]

    #             # Calculate cosine similarity between I_k and mean_i for weighting
    #             img_quality_weight = torch.nn.functional.cosine_similarity(
    #                 I_k.unsqueeze(0),
    #                 mean_j.unsqueeze(0)
    #             ).item()

    #             # Calculate similarity and apply weight
    #             similarity = torch.nn.functional.cosine_similarity(
    #                 diff_image_vector.unsqueeze(0),
    #                 diff_text_vector.unsqueeze(0)
    #             ).item()
    #             modality_gap = torch.nn.functional.cosine_similarity(
    #                 class_means[target_class].unsqueeze(0),
    #                 target_text_embedding.unsqueeze(0)
    #             ).item()
    #             print('found', similarity * weights[j], modality_gap)
    #             weighted_similarity = (similarity * weights[j])
    #             similarities.append((weighted_similarity, j))

    #         # Sort similarities and get the lowest 3
    #         sorted_similarities = sorted(similarities, key=lambda x: x[0])
    #         lowest_3_similarities = sorted_similarities
    #         print(f'For class {target_class}, lowest 3 similarities: {lowest_3_similarities}')

    #         # Append the average similarity value to s_k_scores
    #         top_sims = [sim for sim, _ in lowest_3_similarities]
    #         s_k_scores.append(sum(top_sims)/len(top_sims))

    #     # Return the average consistency score S_i for the class
    #     S_i = torch.tensor(s_k_scores).mean().item()
    #     return S_i, C_i_image
    
    ## ZERO SHOT TEMPLATES FOR TEXT EMBEDDINGS
    ## HOW TO PENALIZE OR CHANGE SCORE

    # def calculate_consistency_score_2(image_embeddings, class_means_image, text_embeddings, target_class):
    #     # Convert lists of embeddings to tensors for the target class
    #     target_class_images = torch.stack([torch.tensor(img) for img in image_embeddings[target_class]])
    #     target_class_mean_image = class_means_image[target_class]

    #     # Calculate intra-class compactness (C_i) for the image space
    #     # C_i_image = torch.mean(torch.norm(target_class_images - target_class_mean_image, p=2, dim=1)).item()
    #     C_i_image = torch.mean(torch.tensor([
    #         1 - torch.nn.functional.cosine_similarity(I_k.unsqueeze(0), target_class_mean_image.unsqueeze(0)).item()
    #         for I_k in target_class_images
    #     ])).item()

    #     # Intra-class compactness for text space is zero, since each class has a single text embedding
    #     C_i_text = 0

    #     k = 10
    #     # Calculate inter-class separability (D_i) using cosine distance in the image space
    #     D_i_image_list = []
    #     for j, embedding_j in class_means_image.items():
    #         if j != target_class:
    #             distance = 1 - torch.nn.functional.cosine_similarity(
    #                 target_class_mean_image.unsqueeze(0), embedding_j.unsqueeze(0)
    #             ).item()
    #             D_i_image_list.append((j, distance))

    #     # Sort the distances and select the top k classes with the lowest distances
    #     D_i_image_list_sorted = sorted(D_i_image_list, key=lambda x: x[1])
    #     top_k_classes_image = [j for (j, _) in D_i_image_list_sorted[:k]]
    #     print('image dist:', top_k_classes_image)

    #     # Compute D_i_text as the mean of the top k distances
    #     D_i_image = torch.mean(torch.tensor([dist for (_, dist) in D_i_image_list_sorted])).item()

    #     # Step 1: Compute and store inter-class distances along with class labels
    #     D_i_text_list = []
    #     for j, embedding_j in text_embeddings.items():
    #         if j != target_class:
    #             distance = 1 - torch.nn.functional.cosine_similarity(
    #                 text_embeddings[target_class].unsqueeze(0), embedding_j.unsqueeze(0)
    #             ).item()
    #             D_i_text_list.append((j, distance))

    #     # Sort the distances and select the top k classes with the lowest distances
    #     D_i_text_list_sorted = sorted(D_i_text_list, key=lambda x: x[1])
    #     top_k_classes_text = [j for (j, _) in D_i_text_list_sorted[:k]]
    #     print('text dist:', top_k_classes_text)
    #     dist_list = top_k_classes_image + list(set(top_k_classes_text) - set(top_k_classes_image))
    #     print(len(dist_list), dist_list)

    #     # Compute D_i_text as the mean of the top k distances
    #     D_i_text = torch.mean(torch.tensor([dist for (_, dist) in D_i_text_list_sorted])).item()
        
    #     D_i_image_dict = dict(D_i_image_list)
    #     D_i_text_dict = dict(D_i_text_list)

    #     combined_distances = {
    #         j: (D_i_text_dict[j] + D_i_image_dict[j]) / 2
    #         for j in D_i_text_dict.keys()
    #     }
    #     epsilon = 1e-6
    #     weights = {
    #         j: 1 / (combined_distances[j] + epsilon)
    #         for j in combined_distances.keys()
    #     }
    #     total_weight = sum(weights.values())
    #     normalized_weights = {j: w / total_weight for j, w in weights.items()}


    #     # Step 2: Compute alignment scores and store them with class labels
    #     alignment_scores_list = []
    #     for j, mean_j_image in class_means_image.items():
    #         if j != target_class:
    #             image_diff = target_class_mean_image - mean_j_image
    #             text_diff = text_embeddings[target_class] - text_embeddings[j]

    #             # Cosine similarity between centroid differences in image and text spaces
    #             alignment_score = torch.nn.functional.cosine_similarity(
    #                 image_diff.unsqueeze(0), text_diff.unsqueeze(0)
    #             ).item()
    #             alignment_scores_list.append((j, alignment_score))
        
    #     alignment_scores_dict = dict(alignment_scores_list)

    #     # Step 3: Select alignment scores for the top k classes
    #     alignment_scores_top_k = [
    #         score for (j, score) in alignment_scores_list if j in dist_list
    #     ]

    #     A_i = sum(
    #         alignment_scores_dict[j] * normalized_weights[j]
    #         for j in alignment_scores_dict.keys()
    #     )

    #     # Compute A_i as the mean of the selected alignment scores
    #     # A_i = torch.mean(torch.tensor([score for (j, score) in alignment_scores_list])).item()
    #     # M_i = torch.mean(torch.tensor(sorted(modality_gaps)[:5])).item()

    #     # Calculate final consistency score
    #     print('dist:', D_i_image + D_i_text)
    #     print('align:', A_i)
    #     print('spread/compactness:', C_i_image)
    #     S_i = A_i # / (C_i_image if C_i_image != 0 else 0)
    #     return S_i, C_i_image

    def calculate_classification_margin(image_embedding, text_embedding, other_class_embeddings):
        """
        Calculate the classification margin for an image in a particular class.
        
        Args:
            image_embedding (np.array): The embedding of the image.
            class_text_embedding (np.array): The embedding of the correct class text.
            other_class_embeddings (list of np.array): A list of embeddings for other class texts.
            
        Returns:
            float: The classification margin s_k' for the image in this class.
        """
        # Cosine similarity with the correct class text
        sim_correct_class =  torch.nn.functional.cosine_similarity(image_embedding.reshape(1, -1), text_embedding.reshape(1, -1)).item()
        
        # Cosine similarities with all other class texts
        sim_other_classes = [ torch.nn.functional.cosine_similarity(image_embedding.reshape(1, -1), emb.reshape(1, -1)).item() for emb in other_class_embeddings]
        
        # Find the maximum similarity with any confounding class
        sim_confounding_class = max(sim_other_classes)
        
        # Classification margin
        s_k_prime = sim_correct_class - sim_confounding_class
        return s_k_prime

    def calculate_per_class_classification_margin(image_embeddings, text_embeddings, target_class):
        """
        Calculate the per-class score S' for a set of images in a particular class.
        
        Args:
            image_embeddings (list of np.array): A list of image embeddings for the class.
            class_text_embedding (np.array): The embedding of the correct class text.
            other_class_embeddings (list of np.array): A list of embeddings for other class texts.
            
        Returns:
            float: The per-class score S' for this class.
        """
        target_text_embedding = text_embeddings[target_class]
        other_class_embeddings = [embedding for class_name, embedding in text_embeddings.items() if class_name != target_class]
        # Calculate classification margin for each image in the class
        margins = [calculate_classification_margin(img_emb, target_text_embedding, other_class_embeddings) for img_emb in image_embeddings]
        
        # Average margin for all images in this class
        per_class_score = np.mean(margins)
        return per_class_score



    results = []

    selected_image_embeddings = generated_image_embeddings if use_generated else image_embeddings

    num = 0
    print(text_embeddings.keys())
    # labels, silhouette_scores = get_silhouettes(selected_image_embeddings)
    for class_name in label_to_class.values():
        k = 3
        consistency_score = calculate_consistency_score(selected_image_embeddings, class_means, text_embeddings, class_name, k)
        consistency_score2, compactness = calculate_consistency_score_2(selected_image_embeddings, class_means, text_embeddings, class_name, k)
        classification_margin = calculate_consistency_score_3(selected_image_embeddings, class_means, text_embeddings, class_name, k) # changing to silhouette
        # classification_margin = 0
        zero_shot_accuracy = calculate_zero_shot_accuracy(class_name, selected_image_embeddings)
        print('------------------------------------------------------')
        results.append((consistency_score, classification_margin, zero_shot_accuracy, consistency_score2, compactness))

    data = []
    for class_name, (consistency_score, classification_margin, zero_shot_accuracy, consistency_score2, compactness) in zip(label_to_class.values(), results):
        data.append({
            'class_name': class_name,
            'consistency_score': consistency_score,
            'classification_margin': classification_margin,
            'zero_shot_accuracy': zero_shot_accuracy,
            'consistency_score2': consistency_score2,
            'compactness': compactness
        })

    # Convert data to DataFrame and save as CSV
    df = pd.DataFrame(data)
    if output_dir != None:
        df.to_csv(os.path.join('.', 'results', output_dir, f'{model_name}_{dataset_name}_{generation_tool if use_generated else "real"}.csv'), index=False)
    else:
        df.to_csv(os.path.join('.', 'results', f'{model_name}_{dataset_name}_{generation_tool if use_generated else "real"}.csv'), index=False)
    # plt.figure(figsize=(10, 6))
    # plt.ylim(0.0, 1)
    # # plt.xlim(0, 0.5)
    # consistency_scores, classification_margin, zero_shot_accuracies, consistency_score2, compactness = zip(*results)
    # for i, class_name in enumerate(label_to_class.values()):
    #     plt.annotate(class_name, (consistency_scores[i], zero_shot_accuracies[i]), fontsize=8, ha='right')
    # plt.xlabel("Consistency Score")
    # plt.ylabel("Zero-Shot Accuracy")
    # correlation_real, _ = spearmanr(consistency_scores, zero_shot_accuracies)
    # print(correlation_real)

    # plt.title("Zero-Shot Accuracy vs. Consistency Score for Each Class")
    # plt.show()

    # plt.figure(figsize=(10, 6))
    # consistency_scores, classification_margin, zero_shot_accuracies, consistency_score2, compactness = zip(*results)
    # for i, class_name in enumerate(label_to_class.values()):
    #     plt.annotate(class_name, (classification_margin[i], zero_shot_accuracies[i]), fontsize=8, ha='right')
    # plt.xlabel("Classification Margin")
    # plt.ylabel("Zero-Shot Accuracy")
    # plt.ylim(0, 1)
    # plt.xlim(-0.1, 0.1)
    # plt.title("Zero-Shot Accuracy vs. Consistency Score for Each Class")
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--dataset_link", type=str, required=True)
    parser.add_argument("--generation_tool", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--use_generated", type=int, required=True)
    parser.add_argument("--image_size", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=False)
    parser.add_argument("--local", type=int, required=False)

    args = parser.parse_args()
    
    main(args.dataset_name, args.dataset_link, args.generation_tool, args.model_name, bool(args.use_generated), args.image_size, bool(args.local), args.output_dir)