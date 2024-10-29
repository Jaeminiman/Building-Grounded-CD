import numpy as np
from typing import List, Tuple
import yaml

def load_config(config_path: str) -> dict:
        """Function to load YAML configuration file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Function to normalize a vector."""
    norm = np.linalg.norm(vector, axis=0, keepdims=True)
    return vector / (norm + 1e-8)  # Prevent division by zero by adding a small value


def cosine_similarity(embed1: np.ndarray, embed2: np.ndarray) -> np.ndarray:
    """Function to calculate cosine similarity between two embeddings."""
    embed1 = embed1.astype(np.float32)
    embed2 = embed2.astype(np.float32)
    
    norm_vector1 = normalize_vector(embed1)
    norm_vector2 = normalize_vector(embed2)
    
    cosine_similarity_map = np.sum(norm_vector1 * norm_vector2, axis=0)
    return cosine_similarity_map


def box_cosine_similarity(initial_embed: np.ndarray, new_embed: np.ndarray, boxes: List[List[float]], width: int, height: int, res_embed: int = 64) -> Tuple[List[float], List[np.ndarray]]:
    """Function to calculate cosine similarity for specific box regions."""
    cos_sim = []
    cos_sim_map = []
    for box in boxes:
        cls, x1, y1, x2, y2 = box
        
        x1, x2 = map(lambda p: int(p / width * res_embed), [x1, x2])
        y1, y2 = map(lambda p: int(p / height * res_embed), [y1, y2])

        x1_pad, y1_pad = map(lambda p: max(p - 1, 0), [x1, y1])
        x2_pad = min(x2 + 1, res_embed - 1)
        y2_pad = min(y2 + 1, res_embed - 1)

        embed1 = initial_embed[..., y1_pad:y2_pad, x1_pad:x2_pad]
        embed2 = new_embed[..., y1_pad:y2_pad, x1_pad:x2_pad]
        
        cosine_similarity_map = cosine_similarity(embed1, embed2)        
        logits = np.mean(cosine_similarity_map)
        
        cos_sim.append(logits)
        cos_sim_map.append(cosine_similarity_map)
    return cos_sim, cos_sim_map


def raw_to_sam_scale(coord: List[int], width: int, height: int, res_sam: int = 1024) -> List[int]:
    """Function to convert raw coordinates to SAM scale."""
    x1, y1, x2, y2 = coord
    x1, x2 = map(lambda p: int(p / width * res_sam), [x1, x2])
    y1, y2 = map(lambda p: int(p / height * res_sam), [y1, y2])
    return [x1, y1, x2, y2]