# legacy.py

import numpy as np
from typing import Dict, List
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


class CLIPTextConsistencyScorer:
    def __init__(self, device: str = None, lambda_sil: float = 2.5, top_k_sil: int = 1000, top_k: int = None, num_embed_cutoff: int = None) -> None:
        """
        Initialize the CLIPTextConsistencyScorer.

        Args:
            device (str, optional): The device to use ('cuda' or 'cpu').
                                    Defaults to 'cpu' since NumPy operations are device-agnostic.
            lambda_sil (float, optional): Lambda parameter for silhouette score. Defaults to 2.5.
            top_k_sil (int, optional): Top K similar classes to consider for silhouette score. Defaults to 1000.
            top_k (int, optional): Top K similar classes to consider for text consistency scores. 
                                   If None, consider all classes. Defaults to None.
            num_embed_cutoff (int, optional): Maximum number of descriptive embeddings to use per class.
                                              If None, use all available embeddings. Defaults to None.
        """
        # Device is not needed for NumPy operations, but kept for consistency
        self.device = device if device else "cpu"
        self.lambda_sil = lambda_sil
        self.top_k_sil = top_k_sil
        self.top_k = top_k
        self.num_embed_cutoff = num_embed_cutoff
        self.lambda_one = 1.0

    def compute(
        self,
        standard_embeddings: np.ndarray,
        descriptive_text_embeddings: List[np.ndarray]
    ) -> Dict[str, List[float]]:
        """
        Compute consistency scores based on precomputed text embeddings.

        Args:
            standard_embeddings (np.ndarray): Array of standard text embeddings with shape (C, D).
            descriptive_text_embeddings (List[np.ndarray]):
                List of arrays, each containing descriptive text embeddings for a class with shape (N_c, D).

        Returns:
            Dict[str, List[float]]:
                Dictionary containing text-only consistency scores, classification margin scores,
                compactness separation scores, CS+compactness separation scores, CM+compactness separation scores,
                and silhouette scores per class.
        """
        text_only_consistency_scores = []
        classification_margin_scores = []
        compactness_separation_scores = []
        CS_plus_compactness_separation_scores = []
        CM_plus_compactness_separation_scores = []
        silhouette_scores = []
        silhouette_scores_lambda_one = []

        num_classes, dim = standard_embeddings.shape

        # Normalize standard embeddings
        standard_embeddings_norm = standard_embeddings / np.linalg.norm(standard_embeddings, axis=1, keepdims=True)  # shape (C, D)

        # Compute pairwise cosine similarities between standard embeddings
        standard_cosine_sim = np.dot(standard_embeddings_norm, standard_embeddings_norm.T)  # shape (C, C)
        # Set diagonal to -inf to exclude self-similarity in max computations
        np.fill_diagonal(standard_cosine_sim, -np.inf)

        # Apply num_embed_cutoff and compute mean descriptive embeddings per class and normalize them
        processed_descriptive_text_embeddings = []
        per_class_mean_descriptive_embeddings = []
        
        for embeddings in descriptive_text_embeddings:
            if embeddings.shape[0] > 0:
                # Apply num_embed_cutoff if specified
                if self.num_embed_cutoff is not None and embeddings.shape[0] > self.num_embed_cutoff:
                    embeddings_cutoff = embeddings[:self.num_embed_cutoff]
                else:
                    embeddings_cutoff = embeddings
                processed_descriptive_text_embeddings.append(embeddings_cutoff)
                mean_embedding = np.mean(embeddings_cutoff, axis=0)
            else:
                processed_descriptive_text_embeddings.append(embeddings)
                mean_embedding = np.zeros((dim,))
            per_class_mean_descriptive_embeddings.append(mean_embedding)
            
        per_class_mean_descriptive_embeddings = np.array(per_class_mean_descriptive_embeddings)  # shape (C, D)
        per_class_mean_descriptive_embeddings_norm = per_class_mean_descriptive_embeddings / np.linalg.norm(per_class_mean_descriptive_embeddings, axis=1, keepdims=True)

        # Compute standard_embeddings_diff for T_i - T_j
        standard_embeddings_diff = standard_embeddings_norm[:, np.newaxis, :] - standard_embeddings_norm[np.newaxis, :, :]  # shape (C, C, D)
        # Precompute norms of standard_embeddings_diff
        norms_standard_embeddings_diff = np.linalg.norm(standard_embeddings_diff, axis=2)  # shape (C, C)

        # Iterate over each class
        for c in tqdm(range(num_classes), desc="Computing consistency scores per class"):
            N_c = processed_descriptive_text_embeddings[c].shape[0]
            if N_c == 0:
                # Handle classes with no descriptive embeddings
                text_only_consistency_scores.append(0.0)
                classification_margin_scores.append(0.0)
                compactness_separation_scores.append(0.0)
                CS_plus_compactness_separation_scores.append(0.0)
                CM_plus_compactness_separation_scores.append(0.0)
                silhouette_scores.append(0.0) # Add for silhouette score
                silhouette_scores_lambda_one.append(0.0)
                continue

            # Normalize descriptive embeddings for class c
            descriptive_embeddings_c = processed_descriptive_text_embeddings[c]  # shape (N_c, D)
            descriptive_embeddings_c_norm = descriptive_embeddings_c / np.linalg.norm(descriptive_embeddings_c, axis=1, keepdims=True)  # shape (N_c, D)

            # Mask to exclude class c
            mask = np.ones(num_classes, dtype=bool)
            mask[c] = False

            # Apply top_k filtering if specified
            if self.top_k is not None and self.top_k < np.sum(mask):
                # Find top_k most similar classes to class c
                class_similarities_c = standard_cosine_sim[c, mask]
                sorted_indices = np.argsort(class_similarities_c)[::-1]
                top_k_indices = sorted_indices[:min(self.top_k, len(sorted_indices))]
                
                # Create new mask with only top_k classes
                original_indices = np.arange(num_classes)[mask]
                selected_indices = original_indices[top_k_indices]
                
                # Reset mask and set only selected classes to True
                mask = np.zeros(num_classes, dtype=bool)
                mask[selected_indices] = True

            # Compute s_i^E components
            # Max interclass similarity
            max_interclass_sim = np.max(standard_cosine_sim[c, mask])  # scalar

            # Min intraclass similarity between T_c and T_i^d
            T_c_norm = standard_embeddings_norm[c, :]  # shape (D,)
            cos_sims_intraclass = np.dot(descriptive_embeddings_c_norm, T_c_norm)  # shape (N_c,)
            min_intraclass_sim = np.min(cos_sims_intraclass)  # scalar

            s_i_E = - max_interclass_sim + min_intraclass_sim  # scalar

            # Compute s_k^T and s'_k^T for each descriptive embedding in class c
            s_k_T_list = []
            s_k_T_prime_list = []
            for k in range(N_c):
                T_k_d_norm = descriptive_embeddings_c_norm[k, :]  # shape (D,)

                # Compute T_k^d - \overline{T}_j^d for all j in mask
                diff_T_d = T_k_d_norm - per_class_mean_descriptive_embeddings_norm[mask, :]  # shape (num_selected, D)
                norms_diff_T_d = np.linalg.norm(diff_T_d, axis=1)  # shape (num_selected,)

                # Retrieve T_c - T_j for all j in mask
                diff_T_std = standard_embeddings_diff[c, mask, :]  # shape (num_selected, D)
                norms_diff_T_std = norms_standard_embeddings_diff[c, mask]  # shape (num_selected,)

                # Compute cosine similarities for s_k^T
                dot_products_T = np.sum(diff_T_d * diff_T_std, axis=1)  # shape (num_selected,)
                denom_T = norms_diff_T_d * norms_diff_T_std  # shape (num_selected,)
                # Handle zero denominators
                denom_T = np.where(denom_T == 0, 1e-8, denom_T)
                cos_sims_T = dot_products_T / denom_T  # shape (num_selected,)

                s_k_T = np.min(cos_sims_T) if len(cos_sims_T) > 0 else 0.0  # scalar
                s_k_T_list.append(s_k_T)

                # Compute cosine similarities for s'_k^T
                # Cosine between T_k^d and T_c - T_j
                dot_products_T_prime = np.dot(diff_T_std, T_k_d_norm)  # shape (num_selected,)
                # Since T_k_d_norm is normalized, denom is norms_diff_T_std
                denom_T_prime = norms_diff_T_std  # shape (num_selected,)
                denom_T_prime = np.where(denom_T_prime == 0, 1e-8, denom_T_prime)
                cos_sims_T_prime = dot_products_T_prime / denom_T_prime  # shape (num_selected,)

                s_k_T_prime = np.min(cos_sims_T_prime) if len(cos_sims_T_prime) > 0 else 0.0  # scalar
                s_k_T_prime_list.append(s_k_T_prime)

            # Compute final scores for class c
            # let us make it seperated
            S_i = (1 / N_c) * np.sum(s_k_T_list) # + s_i_E
            S_i_prime = (1 / N_c) * np.sum(s_k_T_prime_list) # + s_i_E

            text_only_consistency_scores.append(S_i)
            classification_margin_scores.append(S_i_prime)
            compactness_separation_scores.append(s_i_E)
            CS_plus_compactness_separation_scores.append(S_i + s_i_E)
            CM_plus_compactness_separation_scores.append(S_i_prime + s_i_E)

            #########################################################################
            # Compute Silhouette Score - Vectorized
            #########################################################################
            target_text_embedding = standard_embeddings_norm[c, :]
            target_class_mean = per_class_mean_descriptive_embeddings_norm[c, :]
            target_descriptive_embeddings = descriptive_embeddings_c_norm

            # -------------------------
            # Compute a(I_k) - Vectorized for all I_k in class c
            dist_Ik_mean_i = 1 - np.dot(target_descriptive_embeddings, target_class_mean) # Cosine distance
            dist_Ik_text_i = 1 - np.dot(target_descriptive_embeddings, target_text_embedding) # Cosine distance
            a_Ik_vector = dist_Ik_mean_i + self.lambda_sil * dist_Ik_text_i # shape (N_c,)
            a_Ik_vector_lambda_one = dist_Ik_mean_i + self.lambda_one * dist_Ik_text_i # shape (N_c,)

            # -------------------------
            # Compute b(I_k) - Vectorized for all I_k in class c
            # Find top_k similar classes (excluding current class c)
            class_similarities_c = standard_cosine_sim[c, mask] # shape (C-1,)
            sorted_indices = np.argsort(class_similarities_c)[::-1] # Indices of classes sorted by similarity
            top_k_indices = sorted_indices[:min(self.top_k_sil, len(sorted_indices))] # Take top k or fewer if less than k classes available
            valid_classes_indices = np.arange(num_classes)[mask][top_k_indices] # Get the actual indices of top_k classes

            if len(valid_classes_indices) > 0:
                b_Ik_list = []
                b_Ik_list_lambda_one = []
                for class_j_idx in valid_classes_indices:
                    mean_j = per_class_mean_descriptive_embeddings_norm[class_j_idx]
                    text_j = standard_embeddings_norm[class_j_idx]

                    dist_Ik_mean_j = 1 - np.dot(target_descriptive_embeddings, mean_j) # Cosine distance, shape (N_c,)
                    dist_Ik_text_j = 1 - np.dot(target_descriptive_embeddings, text_j) # Cosine distance, shape (N_c,)
                    total_dist_j = dist_Ik_mean_j + self.lambda_sil * dist_Ik_text_j # shape (N_c,)
                    total_dist_j_lambda_one = dist_Ik_mean_j + self.lambda_one * dist_Ik_text_j # shape (N_c,)

                    b_Ik_list.append(total_dist_j)
                    b_Ik_list_lambda_one.append(total_dist_j_lambda_one)

                if b_Ik_list:
                    b_Ik_matrix = np.stack(b_Ik_list, axis=1) # shape (N_c, num_valid_classes)
                    b_Ik_vector = np.min(b_Ik_matrix, axis=1) # shape (N_c,)
                    b_Ik_matrix_lambda_one = np.stack(b_Ik_list_lambda_one, axis=1) # shape (N_c, num_valid_classes)
                    b_Ik_vector_lambda_one = np.min(b_Ik_matrix_lambda_one, axis=1) # shape (N_c,)

                else:
                    b_Ik_vector = np.zeros(N_c) # Should not happen if valid_classes_indices is checked.
                    b_Ik_vector_lambda_one = np.zeros(N_c) # Should not happen if valid_classes_indices is checked.
            else:
                b_Ik_vector = np.zeros(N_c) # If no valid classes, b_Ik is 0
                b_Ik_vector_lambda_one = np.zeros(N_c) # If no valid classes, b_Ik is 0


            # -------------------------
            # Multimodal silhouette for this class - Vectorized
            denom_vector = np.maximum(a_Ik_vector, b_Ik_vector) # shape (N_c,)
            sil_value_vector = np.where(denom_vector < 1e-8, 0.0, (b_Ik_vector - a_Ik_vector) / denom_vector) # shape (N_c,)
            S_sil_i = np.mean(sil_value_vector) if sil_value_vector.size > 0 else 0.0 # scalar
            # -------------------------
            # lambda_one
            denom_vector_lambda_one = np.maximum(a_Ik_vector_lambda_one, b_Ik_vector_lambda_one)
            sil_value_vector_lambda_one = np.where(denom_vector_lambda_one < 1e-8, 0.0, (b_Ik_vector_lambda_one - a_Ik_vector_lambda_one) / denom_vector_lambda_one)
            S_sil_i_lambda_one = np.mean(sil_value_vector_lambda_one) if sil_value_vector_lambda_one.size > 0 else 0.0

            silhouette_scores.append(S_sil_i)
            silhouette_scores_lambda_one.append(S_sil_i_lambda_one)



        return {
            "text_only_consistency_scores": text_only_consistency_scores,
            "classification_margin_scores": classification_margin_scores,
            "compactness_separation_scores": compactness_separation_scores,
            "CS_plus_compactness_separation_scores": CS_plus_compactness_separation_scores,
            "CM_plus_compactness_separation_scores": CM_plus_compactness_separation_scores,
            "silhouette_scores": silhouette_scores , # Add silhouette scores to output,
            "silhouette_scores_lambda_one": silhouette_scores_lambda_one # Add silhouette scores to output,
        }