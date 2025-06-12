# main.py
import argparse
import os
import json
from collections import defaultdict
from scipy.stats import spearmanr
from utils import (
    load_config,
    load_model,
    get_dataset,
    CaptionGenerator,
    get_text_embeddings,
    get_image_embeddings,
    compute_zero_shot_accuracy,
    compute_zero_shot_accuracy_on_pseudo_images,
    compare_and_correlate_accuracies,
    knn_classification,
    plot_accuracy_correlation,
    plot_comparison_accuracies,
    plot_knn_vs_actual_accuracies,
    plot_tsne,
    get_image_loader,
    PseudoImageDataset,
    combine_all_data,
    plot_consistency_scores,  # Ensure this is defined in utils.py
    CLIPTextConsistencyScorer,
    plot_consistency_scores,
)

import numpy as np
from torch.utils.data import DataLoader
import torch  # Needed for device check
from tqdm import tqdm

def compute_pseudo_accuracy(descriptive_text_embeddings_per_class, text_embeddings, text_labels, 
                           num_embed_cutoff=None, batch_size=1000):
    """
    Compute pseudo accuracy using descriptive text embeddings as pseudo images.
    
    Args:
        descriptive_text_embeddings_per_class: List of arrays, each containing descriptive embeddings for a class
        text_embeddings: Standard text embeddings (C, D)
        text_labels: Labels corresponding to text_embeddings
        num_embed_cutoff: Maximum number of descriptive embeddings to use per class
        batch_size: Batch size for similarity computation
    
    Returns:
        pseudo_per_class_accuracy: Dict mapping class names to pseudo accuracies
        pseudo_overall_accuracy: Overall pseudo accuracy
        pseudo_class_correct: Dict mapping class names to correct predictions
        pseudo_class_total: Dict mapping class names to total predictions
    """
    # Apply num_embed_cutoff and create pseudo image dataset
    pseudo_images = []
    pseudo_image_labels = []
    
    for class_idx, embeddings in enumerate(descriptive_text_embeddings_per_class):
        class_name = text_labels[class_idx]
        
        if embeddings.shape[0] > 0:
            # Apply cutoff if specified
            if num_embed_cutoff is not None and embeddings.shape[0] > num_embed_cutoff:
                embeddings_to_use = embeddings[:num_embed_cutoff]
            else:
                embeddings_to_use = embeddings
            
            # Add to pseudo image dataset
            for emb in embeddings_to_use:
                pseudo_images.append(emb)
                pseudo_image_labels.append(class_name)
    
    if len(pseudo_images) == 0:
        # Return empty results if no pseudo images
        return {}, 0.0, {}, {}
    
    pseudo_images = np.array(pseudo_images)  # Shape: (N_pseudo, D)
    
    # Normalize embeddings
    pseudo_images_norm = pseudo_images / np.linalg.norm(pseudo_images, axis=1, keepdims=True)
    text_embeddings_norm = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    
    # Compute similarities and predictions
    correct_predictions = 0
    class_correct = {label: 0 for label in text_labels}
    class_total = {label: 0 for label in text_labels}
    
    # Process in batches to avoid memory issues
    num_pseudo = pseudo_images_norm.shape[0]
    
    for start_idx in range(0, num_pseudo, batch_size):
        end_idx = min(start_idx + batch_size, num_pseudo)
        batch_pseudo = pseudo_images_norm[start_idx:end_idx]
        batch_labels = pseudo_image_labels[start_idx:end_idx]
        
        # Compute similarities: batch_pseudo @ text_embeddings_norm.T
        similarities = np.dot(batch_pseudo, text_embeddings_norm.T)  # Shape: (batch_size, C)
        
        # Get predictions (indices of maximum similarity)
        predictions = np.argmax(similarities, axis=1)
        
        # Convert predictions to class names
        predicted_labels = [text_labels[pred_idx] for pred_idx in predictions]
        
        # Count correct predictions
        for true_label, pred_label in zip(batch_labels, predicted_labels):
            class_total[true_label] += 1
            if true_label == pred_label:
                class_correct[true_label] += 1
                correct_predictions += 1
    
    # Calculate per-class accuracies
    pseudo_per_class_accuracy = {}
    for class_name in text_labels:
        if class_total[class_name] > 0:
            pseudo_per_class_accuracy[class_name] = class_correct[class_name] / class_total[class_name]
        else:
            pseudo_per_class_accuracy[class_name] = 0.0
    
    # Calculate overall accuracy
    total_predictions = sum(class_total.values())
    pseudo_overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    return pseudo_per_class_accuracy, pseudo_overall_accuracy, class_correct, class_total

def parse_args():
    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the configuration file.')
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    # Extract experiment parameters
    experiment_params = config['experiment_params']
    batch_size = experiment_params['batch_size']
    knn_batch_size = experiment_params['knn_batch_size']
    num_captions = experiment_params['num_captions']
    tsne_perplexity = experiment_params['tsne_perplexity']
    tsne_n_iter = experiment_params['tsne_n_iter']
    top_k_classes = experiment_params['top_k_classes']
    pseudo_data_flag = experiment_params['pseudo_data']
    save_results_flag = experiment_params['save_results']
    results_dir = experiment_params['results_dir']
    pseudo_images_folder = experiment_params.get('pseudo_images_folder', 'pseudo_images')
    enable_plot = experiment_params.get('enable_plot', False)
    consistency_scorer_config = config.get('consistency_scorer', {})
    consistency_scorer_enabled = consistency_scorer_config.get('enable', False)
    top_k_values = consistency_scorer_config.get('top_k_values', [None]) if consistency_scorer_enabled else [None]
    num_embed_cutoff_values = consistency_scorer_config.get('num_embed_cutoff_values', [None]) if consistency_scorer_enabled else [None]
    gpt_model = config['gpt_model']['name']
    # if cuda is available, use it
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Loop over models and datasets
    for model_cfg in config['models']:
        for dataset_cfg in config['datasets']:
            model_name = model_cfg['name']
            print(f"Loading model: {model_name}")
            model_info = load_model(model_name)
            # processor = model_info['processor']  # Uncomment if needed

            dataset_name = dataset_cfg['name']
            plotfile_path = os.path.join('figures', f"{model_name.replace('/', '_')}_{dataset_name}")
            os.makedirs(plotfile_path, exist_ok=True)
            print(f"\nProcessing dataset: {dataset_name}")
            dataset = get_dataset(dataset_name)
            n_classes = dataset.classes
            n_object = len(n_classes)
            dataset.class_to_idx = {class_name: idx for idx, class_name in enumerate(n_classes)}

            # Generate prompts and captions
            standard_prompts = [f"a photo of a {class_name}" for class_name in n_classes]
            standard_labels = n_classes

            # Initialize CaptionGenerator
            capGenerator = CaptionGenerator(dataset_name=dataset_name, class_names=n_classes, num_captions=num_captions, model = gpt_model, prompt_template=dataset.prompt_template)
            alter_caption_list = []
            labels = []
            print("Generating alternative captions...")
            for class_name in n_classes:
                alter_captions = capGenerator.get_alternative_captions(class_name)
                alter_caption_list.extend(alter_captions)
                labels.extend([class_name] * len(alter_captions))

            # Compute descriptive text embeddings (CxN_c x D)
            print("Computing descriptive text embeddings...")
            descriptive_text_embeddings = get_text_embeddings(alter_caption_list, model_info, device)
            descriptive_text_labels = labels

            # Ensure embeddings are NumPy arrays
            # if isinstance(descriptive_text_embeddings, torch.Tensor):
            #     descriptive_text_embeddings = descriptive_text_embeddings.cpu().numpy()
            # elif isinstance(descriptive_text_embeddings, list):
            #     descriptive_text_embeddings = [e.cpu().numpy() if isinstance(e, torch.Tensor) else e for e in descriptive_text_embeddings]
            # Else, assume already a NumPy array

            # Compute standard text embeddings (CxD)
            print("Computing standard text embeddings...")
            text_embeddings = get_text_embeddings(standard_prompts, model_info, device)
            text_labels = standard_labels

            # Ensure embeddings are NumPy arrays
            # if isinstance(text_embeddings, torch.Tensor):
            #     text_embeddings = text_embeddings.cpu().numpy()
            # elif isinstance(text_embeddings, list):
            #     text_embeddings = [e.cpu().numpy() if isinstance(e, torch.Tensor) else e for e in text_embeddings]
            # Else, assume already a NumPy array


            # Compute image embeddings
            print("Computing image embeddings...")
            class_indices = [dataset.class_to_idx[class_name] for class_name in n_classes]
            image_loader = get_image_loader(dataset, class_indices, batch_size)
            # if   - name: wds_objectnet # too large
            #       - name: wds_sun397 # too large
            # we make fake image embeddings since the dataset is too large
            if dataset_name in ['wds_objectnet', 'wds_sun397']:
                image_embeddings = np.zeros((len(n_classes) * 35, 512))
                image_labels = n_classes * 35
            else:
                image_embeddings, image_labels = get_image_embeddings(image_loader, model_info, device)
                image_labels = [dataset.classes[label] for label in image_labels]

            # Create data entries
            if enable_plot:
                descriptive_text_data = [
                    {'embedding': emb, 'label': label, 'modality': 'text'}
                    for emb, label in zip(descriptive_text_embeddings, descriptive_text_labels)
                ]
                standard_text_data = [
                    {'embedding': emb, 'label': label, 'modality': 'standard'}
                    for emb, label in zip(text_embeddings, text_labels)
                ]
                image_data = [
                    {'embedding': emb, 'label': label, 'modality': 'image'}
                    for emb, label in zip(image_embeddings, image_labels)
                ]
                # Combine all data modalities
                combined_embeddings, combined_labels, combined_modalities = combine_all_data(
                    descriptive_text_data, standard_text_data, image_data, 
                )

            # Organize descriptive embeddings per class for consistency scorer
            descriptive_text_embeddings_per_class = []
            class_to_descriptive_embeddings = defaultdict(list)
            print("Organizing descriptive embeddings per class...")
            for emb, label in tqdm(zip(descriptive_text_embeddings, descriptive_text_labels)):
                class_to_descriptive_embeddings[label].append(emb)
            for class_name in n_classes:
                embeddings = class_to_descriptive_embeddings[class_name]
                if embeddings:
                    descriptive_text_embeddings_per_class.append(np.stack(embeddings))
                else:
                    # Append an empty array if no descriptive embeddings exist for the class
                    descriptive_text_embeddings_per_class.append(np.empty((0, text_embeddings.shape[1])))
                    
            # Compute zero-shot accuracy on real images (once for all top_k values)
            print("Computing zero-shot accuracy on real images...")
            if dataset_name in ['wds_objectnet', 'wds_sun397']:
                if 'siglip' in model_name:
                    csv_model_name = 'siglip'
                elif 'flava' in model_name:
                    csv_model_name = 'flava'
                else:
                    csv_model_name = 'clip'
                import pandas as pd
                df = pd.read_csv(f'supplement_data/{csv_model_name}_{dataset_name.replace("wds_", "")}.csv')
                # note that real_per_class_accuracy is dictionary from class name to accuracy
                real_per_class_accuracy = {}
                for i,class_name in enumerate(n_classes):
                    real_per_class_accuracy[class_name] = df['zero_shot_accuracy'].iloc[i]
                real_overall_accuracy = df['zero_shot_accuracy'].mean()
                # make up the rest of the real_class_correct and real_class_total
                real_class_correct = {class_name: 0 for class_name in n_classes}
                real_class_total = {class_name: 0 for class_name in n_classes}
                for class_name in n_classes:
                    real_class_correct[class_name] = real_per_class_accuracy[class_name]
                    real_class_total[class_name] = 1
                
            else:   
                real_per_class_accuracy, real_overall_accuracy, real_class_correct, real_class_total = compute_zero_shot_accuracy(
                    image_embeddings=image_embeddings,
                    image_labels=image_labels,
                    text_embeddings=text_embeddings,
                    text_labels=text_labels,
                    batch_size=knn_batch_size,
                )

            # Display results
            print(f"\nPer-Class Zero-Shot Accuracy on Real Images (Using Alternative Captions):")
            for class_name in n_classes:
                accuracy = real_per_class_accuracy.get(class_name)
                if accuracy is not None:
                    print(f"{class_name}: {accuracy:.2%} ({real_class_correct[class_name]}/{real_class_total[class_name]})")
                else:
                    print(f"{class_name}: No samples")
            print(f"\nOverall Zero-Shot Accuracy on Real Images: {real_overall_accuracy:.2%}")

            # KNN Classification on descriptive_text_embeddings (once for all top_k values)
            print("Performing KNN classification...")
            y_test, prob_correct_class, label_encoder = knn_classification(descriptive_text_embeddings, descriptive_text_labels)
            labels_array = np.array(descriptive_text_labels)
            y = label_encoder.transform(labels_array)

            # Initialize dictionaries to accumulate probabilities per class
            class_prob_sums = defaultdict(float)
            class_counts = defaultdict(int)

            # Accumulate probabilities
            for i in range(len(y_test)):
                true_class_idx = y_test[i]
                true_class_label = label_encoder.inverse_transform([true_class_idx])[0]
                class_prob_sums[true_class_label] += prob_correct_class[i]
                class_counts[true_class_label] += 1

            # Compute per-class average probabilities
            knn_per_class_probabilities = {}
            for class_label in class_prob_sums:
                knn_per_class_probabilities[class_label] = class_prob_sums[class_label] / class_counts[class_label]

            # Display per-class average probabilities
            print("\nPer-Class Average Probability Assigned to Correct Class:")
            for class_name in n_classes:
                avg_prob = knn_per_class_probabilities.get(class_name)
                if avg_prob is not None:
                    print(f"{class_name}: {avg_prob:.2%} ({class_counts[class_name]} samples)")
                else:
                    print(f"{class_name}: No samples")

            # Compute overall average probability
            total_prob = sum(class_prob_sums.values())
            total_samples = sum(class_counts.values())
            overall_avg_prob = total_prob / total_samples if total_samples > 0 else 0
            print(f"\nOverall Average Probability Assigned to Correct Class: {overall_avg_prob:.2%}")

            # Collect results for different parameter settings
            all_results = {}
            
            # First loop: vary top_k while keeping num_embed_cutoff=None (use all embeddings)
            print("\n=== Varying top_k values (num_embed_cutoff=all) ===")
            for top_k in top_k_values:
                top_k_str = str(top_k) if top_k is not None else "all"
                print(f"\n--- Processing with top_k={top_k_str}, num_embed_cutoff=all ---")
                
                # Compute consistency scores if enabled
                consistency_scores = None
                if consistency_scorer_enabled:
                    print(f"Initializing CLIPTextConsistencyScorer with top_k={top_k_str}, num_embed_cutoff=all...")
                    consistency_scorer = CLIPTextConsistencyScorer(device=device, top_k=top_k, num_embed_cutoff=None)
                    print("Computing consistency scores...")
                    consistency_scores = consistency_scorer.compute(
                        standard_embeddings=text_embeddings,  # Shape: (C, D)
                        descriptive_text_embeddings=descriptive_text_embeddings_per_class  # List of C tensors, each (N_c, D)
                    )
                    # check consistency scores shape
                    for k in consistency_scores:
                        print(f"Consistency scores shape for {k}: {len(consistency_scores[k])}")
                        
                # Plot only for first top_k value to avoid redundant plots
                if enable_plot and top_k == top_k_values[0]:
                # Plot KNN average probabilities vs actual accuracies
                    plot_knn_vs_actual_accuracies(
                        knn_per_class_probabilities, real_per_class_accuracy, n_classes,
                        title=f'KNN on {dataset_name}',
                        save_path=plotfile_path
                    )
                    # Plot comparison of accuracies
                    if consistency_scorer_enabled and consistency_scores:
                        print("Plotting consistency scores vs actual accuracies...")
                        plot_title = f"Consistency Scores vs Actual Accuracies for {dataset_name}"
                        plot_consistency_scores(
                            consistency_scores=consistency_scores,
                            real_per_class_accuracy=real_per_class_accuracy,
                            n_classes=n_classes,
                            title=plot_title,
                            save_path=plotfile_path
                        )
                    # t-SNE Visualization
                    print("Generating t-SNE visualization...")
                    selected_modalities = ['text', 'standard', 'image',]
                    plot_tsne(
                        combined_embeddings=combined_embeddings,
                        combined_labels=combined_labels,
                        combined_modalities=combined_modalities,
                        per_class_accuracy=real_per_class_accuracy,
                        n_classes=n_classes,
                        top_k=top_k_classes,
                        selected_modalities=selected_modalities,
                        tsne_perplexity=tsne_perplexity,
                        tsne_n_iter=tsne_n_iter,
                        dataset_name=dataset_name,
                        save_path=plotfile_path
                    )

                # Collect results for this top_k value
                if save_results_flag:
                    real_accuracies = [real_per_class_accuracy[label] for label in n_classes]
                    knn_probabilities = [knn_per_class_probabilities[label] for label in n_classes]
                    mae_between_actual_and_knn_probabilities = np.mean(np.abs(np.array(real_accuracies) - np.array(knn_probabilities)))
                    mae_between_actual_and_constant_probabilities = np.mean(np.abs(np.array(real_accuracies) - 0.5))

                    param_results = {
                        'corr_betweem_actual_accuracies_and_knn_probabilities': spearmanr(real_accuracies, knn_probabilities).correlation,
                        'mae': mae_between_actual_and_knn_probabilities,
                        'mase_baseline': mae_between_actual_and_constant_probabilities,
                        'overall_accuracy': real_overall_accuracy,
                        'top_k': top_k,
                        'num_embed_cutoff': None,
                        # 'per_class_accuracy': real_per_class_accuracy,
                        # 'class_correct': dict(real_class_correct),
                        # 'class_total': dict(real_class_total),
                    }
                    
                    # Assuming real_accuracies is a list of accuracies per class
                    if consistency_scorer_enabled and consistency_scores:
                        for key in consistency_scores:
                            correlation = spearmanr(real_accuracies, consistency_scores[key]).correlation
                            param_results[f'corr_between_{key}_and_accuracy'] = correlation
                            print(f"Correlation between {key} and actual accuracies: {correlation:.4f}")

                        if 'silhouette_scores' in consistency_scores and 'text_only_consistency_scores' in consistency_scores:
                            # make a new compound key
                            compound_score = [x+y for x,y in zip(consistency_scores['silhouette_scores'], consistency_scores['text_only_consistency_scores'])]
                            correlation = spearmanr(real_accuracies, compound_score).correlation
                            param_results['corr_between_silhouette_scores_plus_text_only_consistency_scores_and_accuracy'] = correlation
                            print(f"Correlation between silhouette scores plus text only consistency scores and actual accuracies: {correlation:.4f}")
                    
                    # Store results for this top_k value
                    all_results[f'top_k_{top_k_str}'] = param_results

                # Cleanup consistency scorer for this iteration
                if consistency_scorer_enabled:
                    del consistency_scorer
                    if consistency_scores:
                        del consistency_scores
            
            # Second loop: vary num_embed_cutoff while keeping top_k=None (use all classes)
            print("\n=== Varying num_embed_cutoff values (top_k=all) ===")
            for num_embed_cutoff in num_embed_cutoff_values:
                num_embed_cutoff_str = str(num_embed_cutoff) if num_embed_cutoff is not None else "all"
                print(f"\n--- Processing with top_k=all, num_embed_cutoff={num_embed_cutoff_str} ---")
                
                # Compute pseudo accuracy
                print(f"Computing pseudo accuracy with num_embed_cutoff={num_embed_cutoff_str}...")
                pseudo_per_class_accuracy, pseudo_overall_accuracy, pseudo_class_correct, pseudo_class_total = compute_pseudo_accuracy(
                    descriptive_text_embeddings_per_class=descriptive_text_embeddings_per_class,
                    text_embeddings=text_embeddings,
                    text_labels=text_labels,
                    num_embed_cutoff=num_embed_cutoff,
                    batch_size=knn_batch_size
                )
                
                # Display pseudo accuracy results
                print(f"Pseudo accuracy results:")
                for class_name in n_classes:
                    pseudo_acc = pseudo_per_class_accuracy.get(class_name, 0.0)
                    total_samples = pseudo_class_total.get(class_name, 0)
                    correct_samples = pseudo_class_correct.get(class_name, 0)
                    print(f"  {class_name}: {pseudo_acc:.2%} ({correct_samples}/{total_samples})")
                print(f"Overall pseudo accuracy: {pseudo_overall_accuracy:.2%}")
                
                # Compute correlation between pseudo and real accuracy
                real_accuracies = [real_per_class_accuracy[label] for label in n_classes]
                pseudo_accuracies = [pseudo_per_class_accuracy.get(label, 0.0) for label in n_classes]
                pseudo_real_correlation = spearmanr(pseudo_accuracies, real_accuracies).correlation
                print(f"Correlation between pseudo and real accuracy: {pseudo_real_correlation:.4f}")
                
                # Compute consistency scores if enabled
                consistency_scores = None
                if consistency_scorer_enabled:
                    print(f"Initializing CLIPTextConsistencyScorer with top_k=all, num_embed_cutoff={num_embed_cutoff_str}...")
                    consistency_scorer = CLIPTextConsistencyScorer(device=device, top_k=None, num_embed_cutoff=num_embed_cutoff)
                    print("Computing consistency scores...")
                    consistency_scores = consistency_scorer.compute(
                        standard_embeddings=text_embeddings,  # Shape: (C, D)
                        descriptive_text_embeddings=descriptive_text_embeddings_per_class  # List of C tensors, each (N_c, D)
                    )
                    # check consistency scores shape
                    for k in consistency_scores:
                        print(f"Consistency scores shape for {k}: {len(consistency_scores[k])}")

                # Collect results for this num_embed_cutoff value
                if save_results_flag:
                    knn_probabilities = [knn_per_class_probabilities[label] for label in n_classes]
                    mae_between_actual_and_knn_probabilities = np.mean(np.abs(np.array(real_accuracies) - np.array(knn_probabilities)))
                    mae_between_actual_and_constant_probabilities = np.mean(np.abs(np.array(real_accuracies) - 0.5))

                    param_results = {
                        'corr_betweem_actual_accuracies_and_knn_probabilities': spearmanr(real_accuracies, knn_probabilities).correlation,
                        'mae': mae_between_actual_and_knn_probabilities,
                        'mase_baseline': mae_between_actual_and_constant_probabilities,
                        'overall_accuracy': real_overall_accuracy,
                        'top_k': None,
                        'num_embed_cutoff': num_embed_cutoff,
                        # Pseudo accuracy metrics
                        'pseudo_overall_accuracy': pseudo_overall_accuracy,
                        'corr_between_pseudo_and_real_accuracy': pseudo_real_correlation,
                        # 'per_class_accuracy': real_per_class_accuracy,
                        # 'class_correct': dict(real_class_correct),
                        # 'class_total': dict(real_class_total),
                    }
                    
                    # Assuming real_accuracies is a list of accuracies per class
                    if consistency_scorer_enabled and consistency_scores:
                        for key in consistency_scores:
                            correlation = spearmanr(real_accuracies, consistency_scores[key]).correlation
                            param_results[f'corr_between_{key}_and_accuracy'] = correlation
                            print(f"Correlation between {key} and actual accuracies: {correlation:.4f}")

                        if 'silhouette_scores' in consistency_scores and 'text_only_consistency_scores' in consistency_scores:
                            # make a new compound key
                            compound_score = [x+y for x,y in zip(consistency_scores['silhouette_scores'], consistency_scores['text_only_consistency_scores'])]
                            correlation = spearmanr(real_accuracies, compound_score).correlation
                            param_results['corr_between_silhouette_scores_plus_text_only_consistency_scores_and_accuracy'] = correlation
                            print(f"Correlation between silhouette scores plus text only consistency scores and actual accuracies: {correlation:.4f}")
                    
                    # Store results for this num_embed_cutoff value
                    all_results[f'num_embed_cutoff_{num_embed_cutoff_str}'] = param_results

                # Cleanup consistency scorer for this iteration
                if consistency_scorer_enabled:
                    del consistency_scorer
                    if consistency_scores:
                        del consistency_scores

            # Save all parameter results in a single file
            if save_results_flag and all_results:
                results_filepath = os.path.join(results_dir, f"{model_name.replace('/', '_')}_{gpt_model}_{dataset_name}_results.json")
                with open(results_filepath, 'w') as f:
                    json.dump(all_results, f, indent=4)
                print(f"All parameter results saved to {results_filepath}")

            # Cleanup outside the top_k loop
            import gc
            if enable_plot:
                del combined_embeddings, combined_labels, combined_modalities
            del descriptive_text_embeddings, descriptive_text_labels
            del text_embeddings, text_labels
            del image_embeddings, image_labels
            del real_per_class_accuracy, real_class_correct, real_class_total
            del knn_per_class_probabilities, class_prob_sums, class_counts
            del model_info
            gc.collect()


if __name__ == '__main__':
    main()
