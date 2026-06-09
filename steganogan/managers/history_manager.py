import json
import os
from typing import Any, Dict, List, Optional


class HistoryManager:
    """Manages training history and metrics logging."""

    def __init__(self, log_dir: str) -> None:
        self.log_dir: str             = log_dir
        self.history: List[Dict[str, Any]] = []
        self.metrics_path: Optional[str]   = (
            os.path.join(log_dir, 'metrics.log') if log_dir else None
        )
        self._load_existing_history()

    def _load_existing_history(self) -> None:
        """Load existing metrics from the log file if it exists."""
        if self.metrics_path and os.path.exists(self.metrics_path):
            try:
                with open(self.metrics_path, 'r') as f:
                    self.history = json.load(f)
                    print(f'Loaded {len(self.history)} previous epochs.')
            except (json.JSONDecodeError, IOError) as e:
                print(f'Warning: Could not load existing metrics.log: {e}')
                self.history = []

    def append_and_save(self, metrics: Dict[str, Any]) -> None:
        """Append a metrics dict to history and persist to disk."""
        self.history.append(metrics)
        if self.metrics_path:
            with open(self.metrics_path, 'w') as f:
                json.dump(self.history, f, indent=4)

    def print_metrics(self, metrics: Dict[str, Any]) -> None:
        """Print a formatted summary of the current epoch metrics."""
        print(f"\nEpoch {metrics['epoch']} Metrics:")
        print(
            f"  Train - Enc MSE: {metrics['train.encoder_mse']:.6f}, "
            f"Dec Loss: {metrics['train.decoder_loss']:.4f}, "
            f"Dec Acc: {metrics['train.decoder_acc']:.4f}"
        )
        if 'train.cover_score' in metrics:
            print(
                f"  Train - Cover Score: {metrics['train.cover_score']:.6f}, "
                f"Generated Score: {metrics['train.generated_score']:.6f}"
            )
        print(
            f"  Val   - Enc MSE: {metrics['val.encoder_mse']:.6f}, "
            f"Dec Loss: {metrics['val.decoder_loss']:.4f}, "
            f"Dec Acc: {metrics['val.decoder_acc']:.4f}"
        )
        print(
            f"  Val   - SSIM: {metrics['val.ssim']:.4f}, "
            f"PSNR: {metrics['val.psnr']:.2f}, "
            f"RSBPP: {metrics['val.rsbpp']:.4f}"
        )
        if 'val.cover_score' in metrics:
            print(
                f"  Val   - Cover Score: {metrics['val.cover_score']:.6f}, "
                f"Generated Score: {metrics['val.generated_score']:.6f}"
            )
