import os
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional, Union, Tuple

import torch
import numpy as np

# Constants
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DIFF_THRESHOLD = 1e-3
DEFAULT_SIMILARITY_THRESHOLD = 0.99
EPSILON = 1e-5

def setup_logging(name="moe_test", level=logging.INFO):
    """
    Configure logging with consistent formatting.
    
    Args:
        name: Logger name
        level: Logging level
    
    Returns:
        Logger instance
    """
    logging.basicConfig(level=level, format=LOG_FORMAT)
    return logging.getLogger(name)


class ResultCollector:
    """Class for collecting and saving intermediate computation results - supports direct use without initialization"""

    @staticmethod
    def save(name, tensor, output_dir="./moe_results", prefix="", rank=0, logger=None):
        """
        Save intermediate results - static method, can be called directly
        
        Args:
            name (str): Result name
            tensor (torch.Tensor): Tensor to save
            output_dir (str): Output directory
            prefix (str): Filename prefix
            rank (int): Current process rank
            logger: Optional logger instance
        """
        if logger is None:
            logger = setup_logging("result_collector")
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create unique name
        filename = f"{prefix}_{name}_rank{rank}.pt"
        filepath = os.path.join(output_dir, filename)
        
        # Convert to CPU and save
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu()
            
        # Save to dictionary and file
        key = f"{prefix}_{name}_rank{rank}"
        torch.save(tensor, filepath)
        return tensor


class ResultAnalyzer:
    def __init__(
        self,
        results_dir: str = "./moe_results",
        model1_prefix: str = "all2all",
        model2_prefix: str = "epmoe",
        rank: int = 0,
        logger = None
    ):
        """
        Initialize result analyzer
        
        Args:
            results_dir: Directory storing intermediate results
            model1_prefix: Prefix for the first model
            model2_prefix: Prefix for the second model
            rank: Process rank to analyze
            logger: Logger instance
        """
        self.results_dir = Path(results_dir)
        self.model1_prefix = model1_prefix
        self.model2_prefix = model2_prefix
        self.rank = rank
        self.logger = logger or setup_logging("result_analyzer")
        
        # Store loaded results
        self.results: Dict[str, Dict[str, torch.Tensor]] = {}
        
        # Store analysis results
        self.analysis_results: Dict[str, Dict[str, Any]] = {}
        
    def load_results(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Load saved result files
        
        Returns:
            Dictionary of results grouped by prefix
        """
        results = defaultdict(dict)
        
        # Iterate through all pt files in directory
        for file_path in self.results_dir.glob(f"*_rank{self.rank}.pt"):
            filename = file_path.name
            parts = filename.split("_")
            
            # Parse filename to get prefix and variable name
            if len(parts) >= 3 and filename.endswith(f"_rank{self.rank}.pt"):
                prefix = parts[0]  # all2all or epmoe
                # Variable name excludes rank part and prefix
                var_name = "_".join(parts[1:-1])  
                
                # Load tensor
                try:
                    tensor = torch.load(file_path)
                    results[prefix][var_name] = tensor
                    self.logger.info(f"Loaded: {prefix}/{var_name}, Shape: {tensor.shape if isinstance(tensor, torch.Tensor) else 'Not a tensor'}")
                except Exception as e:
                    self.logger.error(f"Failed to load {file_path}: {e}")
        
        self.results = dict(results)
        return self.results
    
    
    @staticmethod
    def compare_tensors(
        tensor1: torch.Tensor, 
        tensor2: torch.Tensor,
        var_name: Optional[str] = None,
        rank: int = 0, 
        verbose: bool = True,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        diff_threshold: float = DEFAULT_DIFF_THRESHOLD,
        logger = None
    ) -> dict:
        """
        Compare outputs from two different implementations and print statistics.
        
        Args:
            tensor1: First tensor to compare
            tensor2: Second tensor to compare
            var_name: Variable name for reporting
            rank: Current process rank for distributed training
            verbose: Whether to print detailed comparison information
            similarity_threshold: Threshold to highlight low similarity channels
            diff_threshold: Threshold for determining numerical equivalence
            logger: Logger instance
            
        Returns:
            Dictionary containing comparison metrics
        """
        if logger is None:
            logger = setup_logging("tensor_metrics")
            
        if var_name:
            logger.info(f"Rank {rank}: Analyzing variable {var_name}")

        # Reshape for token-level cosine similarity
        reshaped_all_gather = tensor1.reshape(-1, tensor1.shape[-1])
        reshaped_all_to_all = tensor2.reshape(-1, tensor2.shape[-1])
            
        # Calculate absolute difference between outputs
        abs_diff = torch.abs(reshaped_all_gather - reshaped_all_to_all)
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()
        
        # Calculate relative difference
        max_abs_values = torch.maximum(torch.abs(reshaped_all_gather), torch.abs(reshaped_all_to_all))
        eps = EPSILON
        rel_diff = abs_diff / (max_abs_values + eps)
        max_rel_diff = rel_diff.max().item()
        mean_rel_diff = rel_diff.mean().item()
        
        # # Reshape for token-level cosine similarity
        # reshaped_all_gather = tensor1.reshape(-1, tensor1.shape[-1])
        # reshaped_all_to_all = tensor2.reshape(-1, tensor2.shape[-1])
        
        # Calculate cosine similarity for each token vector
        token_cos_sim = torch.nn.functional.cosine_similarity(reshaped_all_gather, reshaped_all_to_all, dim=1)
        
        # Find highest and lowest similarity
        max_sim_idx = token_cos_sim.argmax().item()
        min_sim_idx = token_cos_sim.argmin().item()
        max_sim_val = token_cos_sim[max_sim_idx].item()
        min_sim_val = token_cos_sim[min_sim_idx].item()
        avg_sim_val = token_cos_sim.mean().item()
        
        # Determine if outputs are numerically equivalent
        is_equivalent = mean_diff < diff_threshold and max_diff < diff_threshold*10 and mean_rel_diff < diff_threshold*10
        
        # Create results dictionary
        results = {
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "max_rel_diff": max_rel_diff,
            "mean_rel_diff": mean_rel_diff,
            "avg_sim_val": avg_sim_val,
            "min_sim_val": min_sim_val,
            "max_sim_val": max_sim_val,
            "is_equivalent": is_equivalent
        }
        
        if verbose:
            logger.info(f"Rank {rank}: Results comparison:")
            logger.info(f"  Maximum absolute difference: {max_diff:.8f}")
            logger.info(f"  Mean absolute difference: {mean_diff:.8f}")
            logger.info(f"  Maximum relative difference: {max_rel_diff:.8f}")
            logger.info(f"  Mean relative difference: {mean_rel_diff:.8f}")
            
            # Check if outputs are numerically close
            if is_equivalent:
                logger.info(f"Rank {rank}: ✅ Outputs are numerically equivalent (differences likely due to floating point precision)")
            else:
                logger.info(f"Rank {rank}: ❌ Outputs show significant differences!")

            # Output token similarity statistics
            logger.info(f"  Token cosine similarity: avg={avg_sim_val:.8f}")
            logger.info(f"  Highest similarity token: idx={max_sim_idx}, value={max_sim_val:.8f}")
            logger.info(f"  Lowest similarity token: idx={min_sim_idx}, value={min_sim_val:.8f}")

            # Find and display the 10 lowest similarity tokens
            lowest_sim_values, lowest_sim_indices = torch.topk(token_cos_sim, 10, largest=False)
            low_sim_count = 0
            for i, (idx, sim) in enumerate(zip(lowest_sim_indices, lowest_sim_values)):
                if sim < similarity_threshold:
                    low_sim_count += 1
                    logger.info(f"  #{i+1}: Token {idx.item()}, Similarity: {sim.item():.8f}")
                    if low_sim_count <= 3:  # Only show values for first 3 low similarity tokens
                        logger.info(f"    All-gather: {reshaped_all_gather[idx][:5].cpu().numpy()}...")
                        logger.info(f"    All-to-all: {reshaped_all_to_all[idx][:5].cpu().numpy()}...")
            
            if low_sim_count == 0:
                logger.info(f"  All tokens have similarity >= {similarity_threshold}")
        
        return results
    
    def analyze_saved_tensors(self, var_name: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Analyze differences for one or all variables common to both models
        
        Args:
            var_name: Name of specific variable to analyze, or None to analyze all
                
        Returns:
            Dictionary of analysis results
        """
        # Load results if not already loaded
        if not self.results:
            self.load_results()
            
        # Check if model results exist
        if self.model1_prefix not in self.results or self.model2_prefix not in self.results:
            self.logger.warning(f"Model results not found ({self.model1_prefix} or {self.model2_prefix})")
            return {"error": f"Model results not found, please call load_results() first"}
        
        # Get common variables between both models
        common_vars = set(self.results[self.model1_prefix].keys()) & set(self.results[self.model2_prefix].keys())
        variables_to_analyze = sorted(common_vars)
        
        # Perform analysis
        for var in variables_to_analyze:
            # Get tensors from both models
            tensor1 = self.results[self.model1_prefix][var]
            tensor2 = self.results[self.model2_prefix][var]
            
            # Calculate metrics
            metrics = self.compare_tensors(tensor1, tensor2, var_name=var, rank=self.rank, logger=self.logger)
            self.analysis_results[var] = metrics
            
        return self.analysis_results


if __name__ == "__main__":
    # Setup logger
    logger = setup_logging("result_analyzer_main")
    
    # Usage example
    logger.info("Starting analysis...")
    analyzer = ResultAnalyzer(
        results_dir="./moe_results", 
        model1_prefix="all2all", 
        model2_prefix="epmoe",
        rank=0,
        logger=logger
    )
    analyzer.analyze_saved_tensors()
    logger.info("Analysis complete")