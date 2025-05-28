import csv
import io
import time
import uuid
import asyncio
import sys
import logging
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.optim as optim
from starlette.applications import Starlette
from starlette.responses import HTMLResponse, RedirectResponse, Response
from starlette.routing import Route
import base64
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from io import BytesIO
from typing import Dict, Any
import pandas as pd


# Add this custom exception after the imports
class DataValidationError(Exception):
    """Custom exception for data validation errors"""

    pass


# Configuration - hardcoded values (no environment variables)
class Config:
    # Data paths
    SAMPLES_FILE = "refs.csv"
    FEATURES_FILE = "features.npy"
    MODEL_CHECKPOINT = "model_checkpoint.pt"
    STRICT_MODE = True  # Enable strict validation mode

    # Class configuration
    CLASS_NAMES = ["Positive", "Negative", "Other", "Bad Quality"]

    # Model parameters
    MODEL_N_ITER = 10
    MODEL_LEARNING_RATE = 0.1
    MODEL_WEIGHT_DECAY = 0.01

    # Active learning parameters
    SCORE_INTERVAL = 8
    RETRAIN_INTERVAL = 4
    SCORING_METHOD = "weighted_least_confidence"

    # Cache settings
    IMAGE_CACHE_SIZE = 20

    # Server settings
    SAVE_INTERVAL = 300  # seconds
    THREAD_POOL_SIZE = 2

    # UI settings
    IMAGE_SIZE = (512, 512)
    CROP_SIZE = 256


# Logging setup
def setup_logging(log_level="INFO", log_file=None):
    """Setup logging configuration"""
    # Create logs directory if it doesn't exist
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    # Configure logging
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper()), format=log_format, handlers=handlers
    )

    return logging.getLogger(__name__)


# Setup logging
logger = setup_logging("INFO", "logs/app.log")

# Create a thread pool for background tasks
executor = ThreadPoolExecutor(max_workers=Config.THREAD_POOL_SIZE)


def parse_bbox(bbox_str: str) -> list:
    """Parse bbox string into a list of integers"""
    return [int(x) for x in bbox_str.split("-")]


class DataValidator:
    def _validate_features_file(self) -> np.ndarray:
        """Load and validate the features file. Must be 2D and have enough rows for all feat_idx in CSV."""
        try:
            features = np.load(self.features_path)
        except Exception as e:
            raise DataValidationError(f"Could not load features file: {e}")
        if features.ndim != 2:
            raise DataValidationError(f"Features file must be a 2D matrix, got shape {features.shape}")
        # Check that the file has enough rows for all feat_idx in the CSV
        try:
            df = pd.read_csv(self.csv_path)
            if 'feat_idx' not in df.columns:
                raise DataValidationError("CSV missing 'feat_idx' column for feature index validation")
            max_idx = pd.to_numeric(df['feat_idx'], errors='coerce').max()
            if np.isnan(max_idx):
                raise DataValidationError("No valid feat_idx values found in CSV for feature index validation")
            if features.shape[0] < max_idx + 1:
                raise DataValidationError(f"Features file has {features.shape[0]} rows, but max feat_idx in CSV is {max_idx}")
        except Exception as e:
            raise DataValidationError(f"Error validating features file against CSV: {e}")
        return features
    def _validate_annotation_class(self, df: pd.DataFrame):
        """Check that annotation column is either empty (missing in CSV) or a valid class name from Config.CLASS_NAMES. Only empty (not space, 'null', 'None') is allowed for missing."""
        valid_classes = set(Config.CLASS_NAMES)
        invalid = []
        for idx, val in df["annotation"].items():
            # Only allow empty string (not space, not None, not 'null', not 'None')
            if (val == "") or (isinstance(val, float) and np.isnan(val)):
                continue  # allow empty (missing in CSV)
            if not (isinstance(val, str) and val in valid_classes):
                invalid.append((idx, val))
        if invalid:
            self.errors.append(f"Invalid annotation class names at rows: {invalid}")

    """Comprehensive validator for CSV + features file data structure"""

    def __init__(self, csv_path: str, features_path: str, strict_mode: bool = True):
        self.csv_path = Path(csv_path)
        self.features_path = Path(features_path)
        self.strict_mode = strict_mode
        self.errors = []
        self.warnings = []

    def validate_all(self) -> Dict[str, Any]:
        """Run simplified validation checks and return report"""
        try:
            # 1. File existence checks
            self._check_file_existence()

            # 2. Load and validate CSV structure
            df = self._validate_csv_structure()

            # 3. Load and validate features file
            features = self._validate_features_file()

            # 4. Check for every row: filename and feat_idx
            self._validate_row_presence(df)

            # 5. Check for duplicate filenames and feat_idx
            self._validate_duplicates(df)

            # 6. Check that all feat_idx exist in features
            self._validate_feat_idx_range(df, features)

            # 7. Check bbox format (now only empty cell or valid format)
            self._validate_bbox_format(df)

            # 8. Check annotation is valid class name or empty
            self._validate_annotation_class(df)

            # 9. Check all images exist
            self._validate_image_files_exist(df)

            return {
                "valid": len(self.errors) == 0,
                "errors": self.errors,
                "warnings": self.warnings,
                "dataframe": df,
                "features": features,
            }
        except Exception as e:
            self.errors.append(f"Critical validation error: {str(e)}")
            return {
                "valid": False,
                "errors": self.errors,
                "warnings": self.warnings,
                "dataframe": None,
                "features": None,
            }

    def _check_file_existence(self):
        """Check if required files exist"""
        if not self.csv_path.exists():
            raise DataValidationError(f"CSV file not found: {self.csv_path}")

        if not self.features_path.exists():
            raise DataValidationError(f"Features file not found: {self.features_path}")

        # Check file permissions
        if not self.csv_path.is_file():
            raise DataValidationError(f"CSV path is not a file: {self.csv_path}")

        if not self.features_path.is_file():
            raise DataValidationError(
                f"Features path is not a file: {self.features_path}"
            )

    def _validate_csv_structure(self) -> pd.DataFrame:
        """Validate CSV file structure and required columns"""
        try:
            df = pd.read_csv(self.csv_path)
        except pd.errors.EmptyDataError:
            raise DataValidationError("CSV file is empty")
        except pd.errors.ParserError as e:
            raise DataValidationError(f"CSV parsing error: {str(e)}")

        required_cols = ["filename", "bbox", "annotation", "feat_idx"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise DataValidationError(f"Missing required columns: {missing_cols}")

        if len(df) == 0:
            raise DataValidationError("CSV file contains no data rows")

        return df

    def _validate_row_presence(self, df: pd.DataFrame):
        """Check that every row has a filename and feat_idx, and that bbox/annotation are either empty (missing in CSV) or valid."""
        # Check filename
        missing_filename = df["filename"].isna() | (df["filename"] == "")
        if missing_filename.any():
            self.errors.append(f"Rows with missing filename: {missing_filename.sum()}")

        # Check feat_idx
        missing_feat_idx = df["feat_idx"].isna()
        if missing_feat_idx.any():
            self.errors.append(f"Rows with missing feat_idx: {missing_feat_idx.sum()}")

    def _validate_duplicates(self, df: pd.DataFrame):
        """Check for duplicate filenames and feat_idx"""
        dup_files = df["filename"].duplicated(keep=False)
        if dup_files.any():
            self.errors.append(
                f"Duplicate filenames found: {df.loc[dup_files, 'filename'].unique().tolist()}"
            )

        dup_feat_idx = df["feat_idx"].duplicated(keep=False)
        if dup_feat_idx.any():
            self.errors.append(
                f"Duplicate feat_idx found: {df.loc[dup_feat_idx, 'feat_idx'].unique().tolist()}"
            )

    def _validate_feat_idx_range(self, df: pd.DataFrame, features: np.ndarray):
        """Check that all feat_idx exist in features (within range)"""
        feat_idx = pd.to_numeric(df["feat_idx"], errors="coerce")
        if feat_idx.isna().any():
            self.errors.append("Non-numeric feat_idx values found.")
            return
        min_idx = feat_idx.min()
        max_idx = feat_idx.max()
        n_samples = features.shape[0]
        if min_idx < 0 or max_idx >= n_samples:
            self.errors.append(
                f"feat_idx values out of range: min={min_idx}, max={max_idx}, features rows={n_samples}"
            )

    def _validate_bbox_format(self, df: pd.DataFrame):
        """Check bbox format: 'top_row-left_col-bottom_row-right_col' or empty cell (missing in CSV). Only empty (not space, 'null', 'None') is allowed for missing."""
        import re

        bbox_pattern = re.compile(
            r"^\s*-?\d+(?:\.\d+)?-\s*-?\d+(?:\.\d+)?-\s*-?\d+(?:\.\d+)?-\s*-?\d+(?:\.\d+)?\s*$"
        )
        for idx, bbox in df["bbox"].items():
            # Only allow empty string (not space, not None, not 'null', not 'None')
            if (bbox == "") or (isinstance(bbox, float) and np.isnan(bbox)):
                continue  # allow empty (missing in CSV)
            if not (isinstance(bbox, str) and bbox_pattern.match(bbox)):
                self.errors.append(f"Row {idx}: bbox format invalid ('{bbox}')")

    def _validate_image_files_exist(self, df: pd.DataFrame):
        """Check that all image files exist"""
        missing = []
        for idx, filename in df["filename"].items():
            if pd.isna(filename) or filename == "":
                continue
            if not Path(filename).exists():
                missing.append(filename)
        if missing:
            self.errors.append(f"Missing image files: {missing}")


    # print_validation_report removed for assertion-based validation


# Add this convenience function after the DataValidator class
def validate_dataset(
    csv_path: str, features_path: str, strict_mode: bool = True
) -> Dict[str, Any]:
    """Convenience function to validate a dataset"""
    validator = DataValidator(csv_path, features_path, strict_mode)
    result = validator.validate_all()
    assert result["valid"], f"Dataset validation failed: {result['errors']}"
    return result


class DataManager:

    def __init__(self):
        logger.info("Loading features and references...")
        df = pd.read_csv(Config.SAMPLES_FILE)
        self.samples = df.to_dict(orient="records")
        self.feats = np.load(Config.FEATURES_FILE)
        self.N = len(self.samples)
        assert max(s["feat_idx"] for s in self.samples) + 1 <= self.feats.shape[0]
        self.labeled_indices = set()
        self.unlabeled_indices = set(range(self.N))
        # Add image caching
        self.image_cache = {}
        logger.info(f"Loaded {self.N} samples with {self.feats.shape[1]} features")

    def get_features(self, idx):
        feat_idx = self.samples[idx]["feat_idx"]
        return self.feats[feat_idx]

    def get_image(self, idx, center_crop=False):
        if idx >= self.N:
            raise IndexError(f"Sample index {idx} out of range (max: {self.N-1})")

        cache_key = f"{idx}_{center_crop}"
        if cache_key in self.image_cache:
            return self.image_cache[cache_key]

        try:
            sample = self.samples[idx]
            img = Image.open(sample["filename"]).convert("RGB")
            bbox_str = sample.get("bbox", "")
            bbox = None
            if isinstance(bbox_str, str) and bbox_str.strip() != "":
                try:
                    bbox = parse_bbox(bbox_str)
                except Exception:
                    bbox = None

            # Draw bounding box if available
            if bbox and len(bbox) == 4:
                draw = ImageDraw.Draw(img)
                r1, c1, r2, c2 = bbox
                draw.rectangle([c1, r1, c2, r2], outline="green", width=3)

                if center_crop:
                    # Center crop around the bounding box
                    cm, rm = (c1 + c2) // 2, (r1 + r2) // 2
                    crop_size = Config.CROP_SIZE
                    img = img.crop(
                        (
                            max(cm - crop_size, 0),
                            max(rm - crop_size, 0),
                            max(cm - crop_size, 0) + 2 * crop_size,
                            max(rm - crop_size, 0) + 2 * crop_size,
                        )
                    )
                else:
                    img = img.resize(Config.IMAGE_SIZE)
            else:
                img = img.resize(Config.IMAGE_SIZE)

            # Manage cache more efficiently
            if len(self.image_cache) >= Config.IMAGE_CACHE_SIZE:
                # Remove oldest entry (FIFO)
                self.image_cache.pop(next(iter(self.image_cache)))
            self.image_cache[cache_key] = img
            return img
        except Exception as e:
            logger.error(f"Error loading image {idx}: {e}")
            # Create error image
            img = Image.new("RGB", Config.IMAGE_SIZE, color=(240, 240, 240))
            draw = ImageDraw.Draw(img)
            draw.text((100, 240), f"Error loading image {idx}", fill=(255, 0, 0))
            return img

    def mark_labeled(self, idx):
        self.labeled_indices.add(idx)
        self.unlabeled_indices.discard(idx)

    def unmark_labeled(self, idx):
        self.labeled_indices.discard(idx)
        self.unlabeled_indices.add(idx)

    def get_unlabeled_indices(self):
        return np.array(list(self.unlabeled_indices))


class LinearClassifier(nn.Module):
    def __init__(self, n_features, num_classes):
        super().__init__()
        self.linear = nn.Linear(n_features, num_classes)

    def get_logits(self, x):
        return self.linear(x)

    def get_probs(self, x):
        return torch.softmax(self.get_logits(x), dim=1)

    def forward(self, x):
        return self.get_logits(x)


class IncrementalModel:
    def __init__(self, n_features, n_iter, use_ckpt):
        self.n_features = n_features
        self.n_iter = n_iter
        self.model = LinearClassifier(n_features, num_classes=len(Config.CLASS_NAMES))
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=Config.MODEL_LEARNING_RATE,
            weight_decay=Config.MODEL_WEIGHT_DECAY,
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.annotated_feats = []
        self.annotated_labels = []
        self.use_ckpt = use_ckpt
        self.training_history = []

        if use_ckpt:
            self._try_load_checkpoint()

    def _try_load_checkpoint(self):
        try:
            checkpoint_path = Path(Config.MODEL_CHECKPOINT)
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                logger.info("Loaded model checkpoint")
        except Exception as e:
            logger.error(f"Could not load model checkpoint: {e}")

    def save_checkpoint(self):
        try:
            checkpoint_path = Path(Config.MODEL_CHECKPOINT)
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                checkpoint_path,
            )
            logger.info(f"Model checkpoint saved at {time.strftime('%H:%M:%S')}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def add_annotation(self, feat, label):
        self.annotated_feats.append(feat)
        self.annotated_labels.append(int(label))

    def set_annotations(self, feats, labels):
        self.annotated_feats = feats
        self.annotated_labels = labels

    def fit(self):
        if not self.annotated_feats:
            return

        logger.info(f"Training model with {len(self.annotated_feats)} samples")
        X = torch.tensor(np.array(self.annotated_feats), dtype=torch.float32)
        y = torch.tensor(np.array(self.annotated_labels), dtype=torch.long).view(-1)

        self.model.train()
        losses = []
        for _ in range(self.n_iter):
            self.optimizer.zero_grad()
            pred_logits = self.model(X)
            loss = self.loss_fn(pred_logits, y)
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
            print("Loss:", loss.item(), end="\r")

        # Update learning rate
        avg_loss = np.mean(losses)
        self.training_history.append(avg_loss)

        # Save the model after training
        if self.use_ckpt:
            self.save_checkpoint()

    def predict_proba(self, X):
        if not self.annotated_feats:
            return np.ones((len(X), len(Config.CLASS_NAMES))) / len(Config.CLASS_NAMES)
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            preds = self.model.get_probs(X_t).numpy()
        return preds


class Orchestrator:
    def __init__(self):
        self.scoring_method = Config.SCORING_METHOD
        self.data_mgr = DataManager()

        # Validate configuration
        assert (
            self.data_mgr.N > Config.SCORE_INTERVAL
        ), f"Not enough samples ({self.data_mgr.N}), need at least {Config.SCORE_INTERVAL} to run active learning, annotate in a txt yourself."

        self.model = IncrementalModel(
            n_features=self.data_mgr.feats.shape[1],
            n_iter=Config.MODEL_N_ITER,
            use_ckpt=Config.MODEL_CHECKPOINT is not None,
        )
        self.use_ckpt = Config.MODEL_CHECKPOINT is not None
        self.annotations = []  # list of (idx, label)
        self.current_sample = None
        self.previous_samples = []
        self.class_counts = [0] * len(Config.CLASS_NAMES)
        self.score_interval = Config.SCORE_INTERVAL
        self.retrain_interval = Config.RETRAIN_INTERVAL
        self.brier_history = []
        self.is_correct_history = []
        self.current_prediction = None

        # Add tracking for skipped samples
        self.skipped_indices = set()
        self.is_review_mode = False
        self.current_review_index = 0
        self.review_indices = []

        # Batch processing
        self.annotations_since_update = 0

        # Prevent duplicate submissions
        self.last_processed_request = None
        self.last_processed_time = 0
        self.processing_lock = False

        # Initialize candidates
        self.candidates = np.random.choice(
            self.data_mgr.N, Config.SCORE_INTERVAL, replace=False
        ).tolist()
        self.current_sample = self.candidates.pop(0)
        self.next_sample = None
        # Load existing annotations from samples file
        self._load_annotations_from_samples()
        self._ensure_next_sample()

    def get_error_chart(self):
        """Generate an error chart with improved styling and dual metrics"""
        if not self.brier_history or len(self.brier_history) < 2:
            return None

        try:
            # Set matplotlib style for better appearance
            fig, ax1 = plt.subplots(figsize=(12, 6))

            brier_errors = np.array(self.brier_history)
            is_correct = np.array(self.is_correct_history)
            indices = np.arange(len(brier_errors))

            # Primary y-axis for Brier score
            ax1.set_xlabel("Annotation Count", fontsize=12)
            ax1.set_ylabel(
                "Brier Score (lower is better)", color="tab:red", fontsize=12
            )
            ax1.set_ylim(0, min(1, np.max(brier_errors) * 1.1))

            # Plot with improved styling
            ax1.scatter(
                indices,
                brier_errors,
                alpha=0.5,
                color="tab:red",
                s=30,
                label="Brier Score",
            )

            # Regular cumulative average
            means = np.cumsum(brier_errors) / np.arange(1, len(brier_errors) + 1)
            ax1.plot(
                indices, means, "r--", linewidth=2, label="Avg Brier Score", alpha=0.7
            )
            ax1.tick_params(axis="y", labelcolor="tab:red")

            # Secondary y-axis for error rate
            ax2 = ax1.twinx()
            ax2.set_ylabel("Error Rate", color="tab:blue", fontsize=12)
            ax2.set_ylim(0, 1)

            # Calculate error rate with threshold
            threshold = 0.25
            error_rate = np.cumsum(~is_correct) / np.arange(1, len(is_correct) + 1)
            ax2.plot(indices, error_rate, "b-", linewidth=3, label=f"Error Rate")
            ax2.tick_params(axis="y", labelcolor="tab:blue")

            # Improved styling
            ax1.grid(True, alpha=0.3)
            plt.title("Model Performance Over Time", fontsize=14, fontweight="bold")

            # Combined legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

            fig.tight_layout()

            # Convert to base64
            buf = BytesIO()
            plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            img_png = buf.getvalue()
            buf.close()

            return base64.b64encode(img_png).decode("utf-8")

        except Exception as e:
            logger.error(f"Error generating chart: {e}")
            return None

    def _load_annotations_from_samples(self):
        loaded = []
        self.class_counts = [0] * len(Config.CLASS_NAMES)
        for idx, sample in enumerate(self.data_mgr.samples):
            class_name = sample.get("annotation", "")
            if isinstance(class_name, str) and class_name.strip() != "" and class_name in Config.CLASS_NAMES:
                label = Config.CLASS_NAMES.index(class_name)
                loaded.append((idx, label))
                self.class_counts[label] += 1
        if len(loaded) > 10:
            old_n_iter = self.model.n_iter
            self.model.n_iter = 100  # Increase iterations for initial training
            self.retrain(loaded)
            self.model.n_iter = old_n_iter  # Restore original iteration count
        else:
            self.retrain(loaded)

    def retrain(self, annotations):
        self.data_mgr.labeled_indices.clear()
        self.data_mgr.unlabeled_indices = set(range(self.data_mgr.N))
        for idx, _ in annotations:
            self.data_mgr.mark_labeled(idx)
        self.annotations = annotations.copy()
        feats = [self.data_mgr.get_features(idx) for (idx, _) in annotations]
        labels = [lbl for (_, lbl) in annotations]
        self.model.set_annotations(feats, labels)
        self.model.fit()
        self.candidates.clear()

    def get_current_sample_index(self):
        return self.current_sample

    def get_next_sample_index(self):
        return self.next_sample

    def handle_annotation(self, idx, label, request_id=None):
        # Enhanced duplicate detection
        current_time = time.time()
        if (
            idx == self.last_processed_request
            and current_time - self.last_processed_time < 1.0  # Increased threshold
        ) or self.processing_lock:
            logger.warning(f"Skipping duplicate annotation for index {idx}")
            return False, False

        try:
            self.processing_lock = True
            logger.info(f"Processing annotation: idx={idx}, label={label}")

            # Check if already annotated
            if idx in {a[0] for a in self.annotations}:
                logger.warning(f"Index {idx} already annotated")
                return False, False

            # Calculate prediction error if we have a prediction
            if self.current_prediction is not None and len(self.annotations) > 0:
                y_true = label
                y_prob = self.current_prediction
                y_true_onehot = np.zeros(len(Config.CLASS_NAMES))
                y_true_onehot[y_true] = 1
                brier = np.mean((y_prob - y_true_onehot) ** 2)
                self.brier_history.append(brier)
                is_correct = int(np.argmax(y_prob) == y_true)
                self.is_correct_history.append(is_correct)
                logger.debug(
                    f"Brier score for sample {idx}: {brier:.4f}, {'right' if is_correct else 'wrong'} prediction"
                )

            # Save annotation in samples file (update in-memory and write to CSV)
            sample = self.data_mgr.samples[idx]
            class_name = Config.CLASS_NAMES[label]
            sample["annotation"] = class_name
            # Write all samples back to the samples file
            pd.DataFrame(self.data_mgr.samples).to_csv(Config.SAMPLES_FILE, index=False)

            # Update internal state
            self.class_counts[label] += 1
            feat = self.data_mgr.get_features(idx)
            self.model.add_annotation(feat, label)

            # Update tracking
            self.annotations_since_update += 1
            should_fit_model = self.annotations_since_update >= self.retrain_interval
            if should_fit_model:
                self.annotations_since_update = 0

            self.data_mgr.mark_labeled(idx)
            self.annotations.append((idx, label))
            self.previous_samples.append(self.current_sample)

            # Update tracking
            self.last_processed_request = idx
            self.last_processed_time = current_time
            self._pick_next_sample()

            logger.info(f"Successfully processed annotation for index {idx}")
            return True, should_fit_model

        except Exception as e:
            logger.error(f"Error processing annotation: {e}")
            return False, False
        finally:
            self.processing_lock = False

    async def async_fit_model(self):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, self.model.fit)

    # Add a new method to skip the current sample
    def skip_current_sample(self):
        if self.current_sample is None:
            return False

        self.skipped_indices.add(self.current_sample)
        logger.info(f"Skipped sample {self.current_sample}")
        self._pick_next_sample()
        return True

    # Add a method to toggle review mode
    def toggle_review_mode(self):
        if self.is_review_mode:
            # Exit review mode
            self.is_review_mode = False
            self._pick_next_sample()
            return "Exited review mode"
        else:
            # Enter review mode if we have annotations
            if not self.annotations:
                return "No annotations to review"
            self.is_review_mode = True
            self.review_indices = [idx for idx, _ in self.annotations]
            self.current_review_index = 0
            self.current_sample = self.review_indices[0]
            return "Entered review mode"

    # Add method to move to next or previous review image
    def next_review_image(self):
        if not self.is_review_mode or not self.review_indices:
            return False

        self.current_review_index = (self.current_review_index + 1) % len(
            self.review_indices
        )
        self.current_sample = self.review_indices[self.current_review_index]
        return True

    def prev_review_image(self):
        if not self.is_review_mode or not self.review_indices:
            return False

        self.current_review_index = (self.current_review_index - 1) % len(
            self.review_indices
        )
        self.current_sample = self.review_indices[self.current_review_index]
        return True

    # Modify the get_current_sample_index method to indicate mode
    def get_current_sample_mode(self):
        if self.current_sample is None:
            return None, "none"

        mode = "review" if self.is_review_mode else "annotation"
        return self.current_sample, mode

    # Update _pick_next_sample to consider skipped samples
    def _pick_next_sample(self):
        if not self.candidates:
            self.refill_candidates()

        # Find first non-skipped candidate
        while self.candidates and self.candidates[0] in self.skipped_indices:
            self.candidates.pop(0)

        # Determine next sample
        if self.candidates:
            self.current_sample = self.candidates.pop(0)
        elif self.skipped_indices:
            # Use a skipped sample as fallback
            next_sample = next(iter(self.skipped_indices))
            self.current_sample = next_sample
            self.skipped_indices.remove(next_sample)
            logger.info(f"Showing previously skipped sample {self.current_sample}")
        else:
            self.current_sample = None

        self._ensure_next_sample()

    def _ensure_next_sample(self):
        if len(self.candidates) < 2:
            self.refill_candidates()
        self.next_sample = self.candidates[0] if self.candidates else None

    def refill_candidates(self):
        # Get unlabeled indices
        unlabeled = self.data_mgr.get_unlabeled_indices()
        if len(unlabeled) == 0:
            logger.info("No more unlabeled samples!")
            return

        # Get predictions for unlabeled samples
        X = self.data_mgr.feats[unlabeled]
        probs = self.model.predict_proba(X)

        # Calculate based on strategy
        if self.scoring_method == "weighted_least_confidence":
            pred_class = np.argmax(probs, axis=1)
            max_prob = np.max(probs, axis=1)
            class_weights = 1 / (np.array(self.class_counts) + 1)
            sample_weights = class_weights[pred_class]
            uncert = 1 - max_prob
            score = uncert * sample_weights
        else:
            raise ValueError(str(self.scoring_method) + " is not a valid score")

        cost = -score

        # Sort and select candidates
        sorted_indices = np.argsort(cost)[: self.score_interval]
        self.candidates = [unlabeled[i] for i in sorted_indices]

        # Double check candidates are actually unlabeled
        self.candidates = [
            idx for idx in self.candidates if idx in self.data_mgr.unlabeled_indices
        ]

        # Fall back to random sampling if needed
        if not self.candidates and len(unlabeled) > 0:
            logger.info("Falling back to random sampling")
            self.candidates = np.random.choice(
                unlabeled, min(self.score_interval, len(unlabeled)), replace=False
            ).tolist()

    def revert_last_annotation(self):
        if not self.annotations or not self.previous_samples:
            return False

        idx, label = self.annotations.pop()
        self.current_sample = self.previous_samples.pop()

        # Update counters properly
        self.class_counts[label] -= 1

        # Remove the last error point if it exists
        if self.brier_history and len(self.brier_history) > 0:
            # Just pop the last error, as our error_history now stores scalar values
            self.brier_history.pop()
            self.is_correct_history.pop()

        # Save all annotations to samples file (overwrite CSV)
        import pandas as pd
        pd.DataFrame(self.data_mgr.samples).to_csv(Config.SAMPLES_FILE, index=False)

        self.data_mgr.unmark_labeled(idx)
        self.retrain(self.annotations)
        return True

    def get_probability_for(self, idx):
        X = self.data_mgr.get_features(idx)[np.newaxis, :]
        probs = self.model.predict_proba(X)
        self.current_prediction = probs[0]  # remove batch dim
        return self.current_prediction


validate_dataset(
    Config.SAMPLES_FILE, Config.FEATURES_FILE, strict_mode=Config.STRICT_MODE
)

# Initialize with configuration
logger.info("Starting the orchestrator...")
orchestrator = Orchestrator()


async def homepage(request):
    idx = orchestrator.get_current_sample_index()
    if idx is None:
        total = orchestrator.data_mgr.N
        labeled = len(orchestrator.data_mgr.labeled_indices)
        class_list = "".join(
            f"<li>{c}: {orchestrator.class_counts[i]}</li>"
            for i, c in enumerate(Config.CLASS_NAMES)
        )
        return HTMLResponse(
            f"""
        <html>
          <head>
            <title>Active Learning Complete</title>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
              body {{ font-family: Arial, sans-serif; max-width: 600px; margin: 40px auto; padding: 20px; }}
              .completion-stats {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }}
            </style>
          </head>
          <body>
            <h1>🎉 All images have been annotated!</h1>
            <div class="completion-stats">
              <p><strong>Total annotations:</strong> {labeled} out of {total}</p>
              <p><strong>Class distribution:</strong></p>
              <ul>{class_list}</ul>
            </div>
            <p><em>To restart, relaunch the application.</em></p>
          </body>
        </html>
        """
        )

    # Generate a unique ID for this request
    request_id = str(uuid.uuid4())

    # Handle different status messages
    message_type = request.query_params.get("message", "")
    status_message = ""

    if message_type == "duplicate":
        status_message = """
        <div style="background-color:#fff3cd;color:#856404;padding:10px;margin:10px 0;border-radius:5px;border:1px solid #ffeeba;">
            This image was already annotated or a duplicate submission was detected. 
            Here's the next image to annotate.
        </div>
        """
    elif message_type == "success":
        status_message = """
        <div style="background-color:#d4edda;color:#155724;padding:10px;margin:10px 0;border-radius:5px;border:1px solid #c3e6cb;">
            Annotation saved successfully!
        </div>
        """

    # ——— new: fetch bbox & path for display ———
    sample = orchestrator.data_mgr.samples[idx]
    img_path = sample["filename"]
    bbox_str = sample.get("bbox", "")
    path_html = f"<p><strong>Image file:</strong> {img_path}</p>"
    bbox_html = ""
    if isinstance(bbox_str, str) and bbox_str.strip() != "":
        try:
            r, c, h, w = parse_bbox(bbox_str)
            bbox_html = (
                f"<p><strong>Bounding‐box:</strong> row={r}, col={c}, "
                f"height={h}, width={w}</p>"
            )
        except Exception:
            bbox_html = f"<p><strong>Bounding‐box:</strong> Invalid format</p>"
    # ————————————————————————————————

    img = orchestrator.data_mgr.get_image(idx, center_crop=True)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_base64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(
        "utf-8"
    )

    # Get predicted probabilities for all classes
    prob_vec = orchestrator.get_probability_for(idx)
    class_counts = orchestrator.class_counts

    # Insert per-class probability table
    prob_table = "<table style='margin:10px 0;'><tr><th>Class</th><th>Probability</th><th>Count</th></tr>"
    for i, cname in enumerate(Config.CLASS_NAMES):
        prob_table += f"<tr><td>{cname}</td><td>{prob_vec[i]:.3f}</td><td>{class_counts[i]}</td></tr>"
    prob_table += "</table>"

    counter_html = (
        "<p>Counts: "
        + ", ".join(
            f"{Config.CLASS_NAMES[i]}={class_counts[i]}"
            for i in range(len(Config.CLASS_NAMES))
        )
        + "</p>"
    )

    # Progress bar
    total_samples = orchestrator.data_mgr.N
    labeled_count = len(orchestrator.data_mgr.labeled_indices)
    progress_html = f"""
    <div style="margin: 10px 0;">
        <div style="background-color: #eee; width: 100%; height: 20px; border-radius: 10px;">
            <div style="background-color: #4CAF50; width: {(labeled_count/total_samples)*100}%; height: 20px; border-radius: 10px; text-align: center; line-height: 20px; color: white;">
                {labeled_count}/{total_samples} ({(labeled_count/total_samples)*100:.1f}%)
            </div>
        </div>
    </div>
    """

    next_idx = orchestrator.get_next_sample_index()
    prefetch_html = (
        f'<img id="prefetch" src="/prefetch?idx={next_idx}" style="display:none;" />'
        if next_idx is not None
        else ""
    )

    # --- Multi-class annotation buttons ---
    # Each button is mapped to a class, with keyboard shortcuts 1,2,3,4...
    class_buttons_html = ""
    for i, cname in enumerate(Config.CLASS_NAMES):
        class_buttons_html += (
            f'<button type="button" id="classButton{i}" onclick="document.getElementById(\'labelValue\').value=\'{i}\';this.form.submit()">'
            f"{cname} ({i+1})</button>"
        )

    script = """
    <script>
    // Prevent multiple rapid submissions
    let processingSubmission = false;
    let lastKeyTime = 0;
    const THROTTLE_TIME = 500; // ms

    document.addEventListener('keydown', function(event) {
        const now = Date.now();
        
        // Skip if we're already processing or if keys are coming too fast
        if (processingSubmission || (now - lastKeyTime < THROTTLE_TIME)) {
            event.preventDefault();
            return;
        }
        
        lastKeyTime = now;
        
        // Multi-class: 1,2,3,4 keys for classes
        let keyToClass = {{
            {",".join([
                f"'{{n}}': {i},'Digit{{n}}': {i},'Numpad{{n}}': {i}"
                .format(n=i+1, i=i)
                for i in range(len(Config.CLASS_NAMES))
            ])}
        }};
        if (event.code in keyToClass) {{
            submitLabel(keyToClass[event.code], 'classButton'+keyToClass[event.code]);
        }}

        // Backspace: revert
        if (event.code === 'Backspace') {{
            event.preventDefault();
            if (!processingSubmission) {{
                processingSubmission = true;
                window.location.href = '/revert';
            }}
        }}

        // Help modal
        if (event.code === 'KeyH' || event.key === '?') {{
            toggleHelp(true);
        }}
        if (event.code === 'Escape') {{
            toggleHelp(false);
        }}

        function submitLabel(value, buttonId) {
            processingSubmission = true;
            document.getElementById(buttonId).classList.add('active');
            document.getElementById('labelValue').value = value;
            document.getElementById('labelForm').submit();
        }
        
        function toggleHelp(show) {
            document.getElementById('helpModal').style.display = show ? 'block' : 'none';
        }
    });

    // Reset the processing flag when the page has completely loaded
    window.addEventListener('load', function() {
    processingSubmission = false;
    });

    document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('labelForm').addEventListener('submit', function() {
        {"".join([f"document.getElementById('classButton{i}').disabled = true;" for i in range(len(Config.CLASS_NAMES))])}
        
        // Show visual feedback
        document.body.style.opacity = '0.7';
        document.body.insertAdjacentHTML('beforeend', 
        '<div id="loading" style="position:fixed;top:0;left:0;width:100%;height:100%;display:flex;justify-content:center;align-items:center;background:rgba(255,255,255,0.5);z-index:1000;">' +
        '<div style="padding:20px;background:white;border-radius:5px;box-shadow:0 0 10px rgba(0,0,0,0.2);">Processing...</div></div>');
    });
    });
    </script>
    """

    help_modal = """
    <div id="helpModal" style="display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.7); z-index: 1000;">
      <div style="background: white; margin: 100px auto; padding: 20px; width: 60%; border-radius: 10px;">
        <h2>Keyboard Shortcuts</h2>
        <ul>
          {''.join([f"<li><strong>{i+1}</strong>: {cname}</li>" for i, cname in enumerate(Config.CLASS_NAMES)])}
          <li><strong>Backspace</strong>: Revert last annotation</li>
          <li><strong>H</strong> or <strong>?</strong>: Show/hide help</li>
          <li><strong>Esc</strong>: Close help</li>
        </ul>
        <button onclick="document.getElementById('helpModal').style.display='none'">Close</button>
      </div>
    </div>
    """

    error_chart_html = ""
    error_chart = orchestrator.get_error_chart()
    if error_chart:
        error_chart_html = f"""
        <div style="margin: 20px 0;">
            <h3>Prediction Error</h3>
            <img src="data:image/png;base64,{error_chart}" style="max-width:100%; height:auto; border:1px solid #ddd;" alt="Error Chart" />
            <p><small>Lower values indicate better model performance. The red line shows the average.</small></p>
        </div>
        """

    html = f"""
    <html>
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Image Annotation Tool</title>
        <style>
          body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
          button {{ padding: 10px 20px; margin: 5px; font-size: 16px; cursor: pointer; }}
          button:disabled {{ opacity: 0.5; cursor: not-allowed; }}
          button.active {{ background-color: #4CAF50; color: white; }}
          img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
        </style>
      </head>
      <body>
        <h1>Sample {idx}</h1>
        {path_html}
        {bbox_html}
        {prob_table}
        {counter_html}
        {progress_html}
        {status_message}
        <img src="{img_base64}" width="512" height="512"/>
        {prefetch_html}
        <form method="POST" action="/label" id="labelForm">
          <input type="hidden" name="idx" value="{idx}"/>
          <input type="hidden" name="request_id" value="{request_id}"/>
          <input type="hidden" name="label" id="labelValue" value=""/>
          {class_buttons_html}
        </form>
        <p>Press <strong>H</strong> or <strong>?</strong> for keyboard shortcuts</p>
        {help_modal}
        {error_chart_html}
        {script}
      </body>
    </html>
    """
    return HTMLResponse(html)


async def prefetch_handler(request):
    try:
        idx = int(request.query_params["idx"])
        img = orchestrator.data_mgr.get_image(idx, center_crop=True)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return Response(buf.getvalue(), media_type="image/jpeg")
    except Exception as e:
        logger.error(f"Error in prefetch: {e}")
        # Return a transparent 1x1 pixel GIF
        return Response(
            b"GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff\x00\x00\x00!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;",
            media_type="image/gif",
        )


async def label_handler(request):
    try:
        form = await request.form()
        idx = int(form["idx"])
        label = int(form["label"])
        request_id = form.get("request_id", str(uuid.uuid4()))

        # Validate input
        assert type(label) is int, "Label must be an integer"
        if label not in range(len(Config.CLASS_NAMES)):
            return HTMLResponse("Invalid label value", status_code=400)

        # Process the annotation with duplicate checking
        success, should_fit_model = orchestrator.handle_annotation(
            idx, label, request_id
        )

        # If duplicate or already annotated, force selection of a new sample
        # before redirecting to homepage
        if not success:
            # Force selection of next sample instead of showing the same one again
            orchestrator._pick_next_sample()
            return RedirectResponse("/?message=duplicate", status_code=303)

        # Start model fitting in the background if needed
        if should_fit_model:
            asyncio.create_task(orchestrator.async_fit_model())

        return RedirectResponse("/?message=success", status_code=303)
    except Exception as e:
        import traceback

        traceback.print_exc()
        return HTMLResponse(f"Error processing annotation: {str(e)}", status_code=500)


async def revert_handler(request):
    try:
        # Apply the same locking mechanism
        if orchestrator.processing_lock:
            return HTMLResponse(
                "Another operation is in progress, please try again", status_code=429
            )

        orchestrator.processing_lock = True
        try:
            success = orchestrator.revert_last_annotation()
            if not success:
                return HTMLResponse("Nothing to revert", status_code=400)
        finally:
            orchestrator.processing_lock = False

        return RedirectResponse("/", status_code=303)
    except Exception as e:
        import traceback

        traceback.print_exc()
        return HTMLResponse(f"Error reverting annotation: {str(e)}", status_code=500)


app = Starlette(
    debug=False,
    routes=[
        Route("/", homepage),
        Route("/label", label_handler, methods=["POST"]),
        Route("/prefetch", prefetch_handler),
        Route("/revert", revert_handler),
    ],
)

# to run locally through vscode
# uvicorn app:app --port 8001

# to expose to the network
# uvicorn app:app --host 0.0.0.0 --port 2333

# the data should be a csv with columns:
# - filename: path to the image file
# - bbox: optional bounding box in the format "row,col,height,width"
# - annotation: optional class name for the image