import torch
import numpy as np
from typing import List

from .utils import fix_random_seed
from sklearn.cluster import KMeans

def random_sampling(pred_probs: torch.Tensor, budget : int = 1000) -> List[int]:
  '''Randomly pick examples.
  :param pred_probs: list of predicted probabilities for the production set in order.
  :param budget: the number of examples you are allowed to pick for labeling.
  :return indices: A list of indices (into the `pred_probs`) for examples to label.
  '''
  fix_random_seed(42)
  
  indices = []
  # ================================
  # FILL ME OUT
  # Randomly pick a 1000 examples to label. This serves as a baseline.
  # Note that we fixed the random seed above. Please do not edit.
  # HINT: when you randomly sample, do not choose duplicates.
  # HINT: please ensure indices is a list of integers
  # ================================

  all_indices = np.arange(len(pred_probs))
  indices = np.random.choice(all_indices, size=budget, replace=False)
  indices = indices.tolist()
  # total_examples = len(pred_probs)
  # indices = random.sample(range(total_examples), budget)

  return indices

def uncertainty_sampling(pred_probs: torch.Tensor, budget : int = 1000) -> List[int]:
  '''Pick examples where the model is the least confident in its predictions.
  :param pred_probs: list of predicted probabilities for the production set in order.
  :param budget: the number of examples you are allowed to pick for labeling.
  :return indices: A list of indices (into the `pred_probs`) for examples to label.
  '''
  indices = []
  chance_prob = 1 / 10.  # may be useful
  # ================================
  # FILL ME OUT
  # Sort indices by the predicted probabilities and choose the 1000 examples with 
  # the least confident predictions. Think carefully about what "least confident" means 
  # for a N-way classification problem.
  # Take the first 1000.
  # HINT: please ensure indices is a list of integers
  # ================================
  
  # Calculate the distance from chance_prob for each probability
  distance_from_chance = torch.abs(pred_probs - chance_prob)
  
  # Get the minimum distance for each example
  min_distances, _ = torch.min(distance_from_chance, dim=1)
  
  # Sort indices by the minimum distances (ascending order)
  sorted_indices = torch.argsort(min_distances)
  
  indices = sorted_indices[:budget].tolist()

  return indices

def margin_sampling(pred_probs: torch.Tensor, budget : int = 1000) -> List[int]:
  '''Pick examples where the difference between the top two predicted probabilities is the smallest.
  :param pred_probs: list of predicted probabilities for the production set in order.
  :param budget: the number of examples you are allowed to pick for labeling.
  :return indices: A list of indices (into the `pred_probs`) for examples to label.
  '''
  indices = []
  # ================================
  # FILL ME OUT
  # Sort indices by the different in predicted probabilities in the top two classes per example.
  # Take the first 1000.
  # ================================
  # Get the top two probabilities for each example
  top_2_probs, _ = torch.topk(pred_probs, k=2, dim=1)
  
  # Calculate the margin (difference between top two probabilities)
  margins = top_2_probs[:, 0] - top_2_probs[:, 1]
  
  # Sort indices by margin (ascending order)
  sorted_indices = torch.argsort(margins)
  
  indices = sorted_indices[:budget].tolist()

  return indices

def entropy_sampling(pred_probs: torch.Tensor, budget : int = 1000) -> List[int]:
  '''Pick examples with the highest entropy in the predicted probabilities.
  :param pred_probs: list of predicted probabilities for the production set in order.
  :param budget: the number of examples you are allowed to pick for labeling.
  :return indices: A list of indices (into the `pred_probs`) for examples to label.
  '''
  indices = []
  epsilon = 1e-6
  # ================================
  # FILL ME OUT
  # Entropy is defined as -E_classes[log p(class | input)] aja the expected log probability
  # over all K classes. See https://en.wikipedia.org/wiki/Entropy_(information_theory).
  # Sort the indices by the entropy of the predicted probabilities from high to low.
  # Take the first 1000.
  # HINT: Add epsilon when taking a log for entropy computation
  # ================================

  # Calculate entropy for each example
  log_probs = torch.log(pred_probs + epsilon)
  entropies = -torch.sum(pred_probs * log_probs, dim=1)
  
  # Sort indices by entropy (descending order)
  sorted_indices = torch.argsort(entropies, descending=True)
  
  indices = sorted_indices[:budget].tolist()

  return indices
