import csv
import io
import time
import uuid
import asyncio
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

# Create a thread pool for background tasks
executor = ThreadPoolExecutor(max_workers=2)

# Configs
REFS_FILE = "instance_references_EXAMPLE.txt"
FEATURES_FILE = "features.npy"
ANNOTATIONS_FILE = "annotations.csv"
MODEL_CHECKPOINT = "model_checkpoint.pt"


class DataManager:
    def __init__(self, refs_file):
        print("Loading features and references...")
        self.samples = []
        with open(refs_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                img_path = parts[0]
                bbox = list(map(int, parts[1:5])) if len(parts) == 5 else None
                self.samples.append({"img_path": img_path, "bbox": bbox})
        self.feats = np.load(FEATURES_FILE)
        self.N = len(self.samples)
        assert self.N == self.feats.shape[0]
        self.labeled_indices = set()
        self.unlabeled_indices = set(range(self.N))
        # Add image caching
        self.image_cache = {}
        self.cache_size = 20  # Maximum number of images to keep in memory

    def get_features(self, idx):
        return self.feats[idx]

    def get_image(self, idx, center_crop=False):
        cache_key = f"{idx}_{center_crop}"
        if cache_key in self.image_cache:
            return self.image_cache[cache_key]

        try:
            sample = self.samples[idx]
            img = Image.open(sample["img_path"]).convert("RGB")
            bbox = sample["bbox"]

            # Draw bounding box if available
            if bbox:
                draw = ImageDraw.Draw(img)
                r1, c1, r2, c2 = bbox
                draw.rectangle([c1, r1, c2, r2], outline="green", width=3)

                if center_crop:
                    # Center crop around the bounding box
                    cm, rm = (c1 + c2) // 2, (r1 + r2) // 2
                    img = img.crop(
                        (
                            max(cm - 256, 0),
                            max(rm - 256, 0),
                            max(cm - 256, 0) + 512,
                            max(rm - 256, 0) + 512,
                        )
                    )
                else:
                    img = img.resize((512, 512))
            else:
                img = img.resize((512, 512))

            # Manage cache
            if len(self.image_cache) >= self.cache_size:
                self.image_cache.pop(next(iter(self.image_cache)))
            self.image_cache[cache_key] = img

            return img
        except Exception as e:
            print(f"Error loading image {idx}: {e}")
            # Create error image
            img = Image.new("RGB", (512, 512), color=(240, 240, 240))
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


class PyTorchLogisticRegression(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


class IncrementalModel:
    def __init__(self, n_features, n_iter=10, use_ckpt=False):
        self.n_features = n_features
        self.n_iter = n_iter
        self.model = PyTorchLogisticRegression(n_features)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, weight_decay=0.01)
        self.loss_fn = nn.BCELoss()
        self.annotated_feats = []
        self.annotated_labels = []
        self.use_ckpt = use_ckpt

        if use_ckpt:
            # Try to load existing model if available
            self._try_load_checkpoint()

    def _try_load_checkpoint(self):
        try:
            checkpoint_path = Path(MODEL_CHECKPOINT)
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                print("Loaded model checkpoint")
        except Exception as e:
            print(f"Could not load model checkpoint: {e}")

    def save_checkpoint(self):
        try:
            checkpoint_path = Path(MODEL_CHECKPOINT)
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                checkpoint_path,
            )
            print(f"Model checkpoint saved at {time.strftime('%H:%M:%S')}")
        except Exception as e:
            print(f"Failed to save model: {e}")

    def add_annotation(self, feat, label):
        self.annotated_feats.append(feat)
        self.annotated_labels.append(int(label))

    def set_annotations(self, feats, labels):
        self.annotated_feats = feats
        self.annotated_labels = labels

    def fit(self):
        if not self.annotated_feats:
            return
        X = torch.tensor(np.array(self.annotated_feats), dtype=torch.float32)
        y = torch.tensor(np.array(self.annotated_labels), dtype=torch.float32).view(
            -1, 1
        )
        self.model.train()
        for _ in range(self.n_iter):
            self.optimizer.zero_grad()
            preds = self.model(X)
            loss = self.loss_fn(preds, y)
            loss.backward()
            self.optimizer.step()
            print("Loss:", loss.item(), end="\r")

        # Save the model after training
        if self.use_ckpt:
            self.save_checkpoint()

    def predict_proba(self, X):
        if not self.annotated_feats:
            return np.ones((len(X), 2)) * 0.5
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            preds = self.model(X_t).numpy().flatten()
        return np.stack([1 - preds, preds], axis=1)


class Orchestrator:
    def __init__(
        self,
        refs_file,
        annotation_file,
        score_interval=8,
        retrain_interval=4,
        n_iter=10,
        sample_cost="certainty",
    ):
        self.sample_cost = sample_cost
        self.data_mgr = DataManager(refs_file)
        assert self.data_mgr.N > score_interval, "Not enough samples, do it by hand"
        self.model = IncrementalModel(
            n_features=self.data_mgr.feats.shape[1], n_iter=n_iter
        )
        self.annotation_file = Path(annotation_file)
        self.annotations = []  # list of (idx, label)
        self.current_sample = None
        self.previous_samples = []
        self.positive_count = 0
        self.negative_count = 0
        self.score_interval = score_interval
        self.error_history = []
        self.current_prediction = None

        # Add tracking for skipped samples
        self.skipped_indices = set()
        self.is_review_mode = False
        self.current_review_index = 0
        self.review_indices = []

        # Batch processing
        self.retrain_interval = retrain_interval  # Update model every N annotations
        self.annotations_since_update = 0

        # Prevent duplicate submissions
        self.last_processed_request = None
        self.last_processed_time = 0
        self.processing_lock = False
        self.last_save_timestamp = time.time()
        self.save_interval = 300  # 5 minutes

        # Initialize candidates
        self.candidates = np.random.choice(
            self.data_mgr.N, score_interval, replace=False
        ).tolist()
        self.current_sample = self.candidates.pop(0)
        self.next_sample = None
        if self.annotation_file.exists():
            self._load_annotations_and_retrain()
        self._ensure_next_sample()

    def get_error_chart(self):
        """Generate an error chart as base64 encoded image with dual y-axes"""
        if not self.error_history or len(self.error_history) < 2:
            return None

        # Create figure with appropriate size
        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Extract data - ensure we're working with numpy arrays for operations
        errors = np.array(self.error_history)
        indices = np.arange(len(errors))

        # Primary y-axis for Brier score (left)
        ax1.set_xlabel("Annotation Count")
        ax1.set_ylabel("Brier Score (lower is better)", color="tab:red")
        ax1.set_ylim(0, 1)  # Brier score range

        # Plot individual points for Brier score
        ax1.scatter(
            indices, errors, alpha=0.5, color="tab:red", s=20, label="Brier Score"
        )

        # Calculate and plot cumulative average
        means = np.cumsum(errors) / np.arange(1, len(errors) + 1)
        ax1.plot(indices, means, "r-", linewidth=2, label="Avg Brier Score")
        ax1.tick_params(axis="y", labelcolor="tab:red")

        # Create second y-axis for error rate (right)
        ax2 = ax1.twinx()
        ax2.set_ylabel("Error Rate", color="tab:blue")
        ax2.set_ylim(0, 1)  # Error rate range

        # Calculate and plot error rate (points with brier score < 0.25)
        error_rate = np.cumsum(errors > 0.25) / np.arange(1, len(errors) + 1)
        print(errors)
        print(error_rate)
        ax2.plot(indices, error_rate, "b-", linewidth=2, label="Error Rate")
        ax2.tick_params(axis="y", labelcolor="tab:blue")

        # Add grid lines for better readability
        ax1.grid(True, alpha=0.3)

        # Add title
        plt.title("Model Performance Over Time")

        # Add combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

        # Tight layout to optimize spacing
        fig.tight_layout()

        # Convert plot to base64 image
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        img_png = buf.getvalue()
        buf.close()

        return base64.b64encode(img_png).decode("utf-8")

    def _load_annotations_and_retrain(self):
        loaded = []
        self.positive_count = 0
        self.negative_count = 0
        with open(self.annotation_file, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                try:
                    idx, label = int(row[0]), int(row[1])
                    loaded.append((idx, label))
                    # Update counters
                    if label == 1:
                        self.positive_count += 1
                    else:
                        self.negative_count += 1
                except (ValueError, IndexError) as e:
                    print(f"Error loading annotation row: {e}")
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
        # Prevent duplicate submissions
        current_time = time.time()
        if (
            idx == self.last_processed_request
            and current_time - self.last_processed_time < 0.5
        ) or self.processing_lock:
            print(
                f"Skipping annotation: {'duplicate request' if idx == self.last_processed_request else 'processing lock active'}"
            )
            return False, False

        try:
            self.processing_lock = True

            # Check if this index is already annotated
            if idx in {a[0] for a in self.annotations}:
                print(f"Index {idx} was already annotated, skipping")
                return False, False

            if self.current_prediction is not None and len(self.annotations) > 0:
                brier = (self.current_prediction - label) ** 2
                self.error_history.append(brier)

            # Regular annotation process
            with open(self.annotation_file, "a", newline="") as f:
                csv.writer(f).writerow([idx, label])

            # Update counters and model
            self.positive_count += 1 if label == 1 else 0
            self.negative_count += 1 if label == 0 else 0

            feat = self.data_mgr.get_features(idx)
            self.model.add_annotation(feat, label)

            # Track annotation and update sample selection
            self.annotations_since_update += 1
            should_fit_model = self.annotations_since_update >= self.retrain_interval
            if should_fit_model:
                self.annotations_since_update = 0

            self.data_mgr.mark_labeled(idx)
            self.annotations.append((idx, label))
            self.previous_samples.append(self.current_sample)

            # Periodic model saving
            current_time = time.time()
            if current_time - self.last_save_timestamp > self.save_interval:
                self.model.save_checkpoint()
                self.last_save_timestamp = current_time

            # Update request tracking
            self.last_processed_request = idx
            self.last_processed_time = current_time
            self._pick_next_sample()

            return True, should_fit_model

        except Exception as e:
            print(f"Error processing annotation: {e}")
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
        print(f"Skipped sample {self.current_sample}")
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
            print(f"Showing previously skipped sample {self.current_sample}")
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
            print("No more unlabeled samples!")
            return

        # Get predictions for unlabeled samples
        X = self.data_mgr.feats[unlabeled]
        probs = self.model.predict_proba(X)

        # Calculate cost based on strategy
        cost = (
            np.abs(probs[:, 1] - 0.5)
            if self.sample_cost == "certainty"
            else probs[:, 0]
        )

        # Sort and select candidates
        sorted_indices = np.argsort(cost)[: self.score_interval]
        self.candidates = [unlabeled[i] for i in sorted_indices]

        # Double check candidates are actually unlabeled
        self.candidates = [
            idx for idx in self.candidates if idx in self.data_mgr.unlabeled_indices
        ]

        # Fall back to random sampling if needed
        if not self.candidates and len(unlabeled) > 0:
            print("Falling back to random sampling")
            self.candidates = np.random.choice(
                unlabeled, min(self.score_interval, len(unlabeled)), replace=False
            ).tolist()

    def revert_last_annotation(self):
        if not self.annotations or not self.previous_samples:
            return False

        idx, label = self.annotations.pop()
        self.current_sample = self.previous_samples.pop()

        # Update counters properly
        if label == 1:
            self.positive_count -= 1
        else:
            self.negative_count -= 1

        # Remove the last error point if it exists
        if self.error_history and len(self.error_history) > 0:
            # Just pop the last error, as our error_history now stores scalar values
            self.error_history.pop()

        with open(self.annotation_file, "w", newline="") as f:
            writer = csv.writer(f)
            for aidx, albl in self.annotations:
                writer.writerow([aidx, albl])

        self.data_mgr.unmark_labeled(idx)
        self.retrain(self.annotations)
        return True

    def get_probability_for(self, idx):
        X = self.data_mgr.get_features(idx)[np.newaxis, :]
        probs = self.model.predict_proba(X)
        self.current_prediction = probs[0, 1]
        return self.current_prediction


print("Starting the orchestrator...")
orchestrator = Orchestrator(
    refs_file=REFS_FILE,
    annotation_file=ANNOTATIONS_FILE,
    score_interval=8,
    n_iter=10,
    sample_cost="certainty",
)


async def homepage(request):
    idx = orchestrator.get_current_sample_index()
    if idx is None:
        total = orchestrator.data_mgr.N
        labeled = len(orchestrator.data_mgr.labeled_indices)
        return HTMLResponse(
            f"""
        <html>
          <head>
            <title>Active Learning Complete</title>
            <style>
              body {{ font-family: Arial, sans-serif; max-width: 600px; margin: 40px auto; padding: 20px; }}
            </style>
          </head>
          <body>
            <h1>All images have been annotated!</h1>
            <p>You have completed {labeled} annotations out of {total} total images.</p>
            <p>Positive annotations: {orchestrator.positive_count}</p>
            <p>Negative annotations: {orchestrator.negative_count}</p>
            <p>To restart, relaunch the app.</p>
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

    img = orchestrator.data_mgr.get_image(idx, center_crop=True)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_base64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(
        "utf-8"
    )

    prob = orchestrator.get_probability_for(idx)
    counter_html = f"<p>Annotations: Positive={orchestrator.positive_count}, Negative={orchestrator.negative_count}</p>"

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
    # Enhanced keyboard handler script

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
        
        // Define keyboard actions mapping
        const actions = {
            'a': () => submitLabel('0', 'negButton'),
            'ArrowLeft': () => submitLabel('0', 'negButton'),
            'd': () => submitLabel('1', 'posButton'),
            'ArrowRight': () => submitLabel('1', 'posButton'),
            'Backspace': () => {
                event.preventDefault();
                if (!processingSubmission) {
                    processingSubmission = true;
                    window.location.href = '/revert';
                }
            },
            'h': () => toggleHelp(true),
            '?': () => toggleHelp(true),
            'Escape': () => toggleHelp(false)
        };
        
        // Execute the action if defined for this key
        if (actions[event.key]) {
            actions[event.key]();
        }
        
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

    // Add submission control to the form
    document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('labelForm').addEventListener('submit', function() {
        // Disable buttons to prevent double submission
        document.getElementById('negButton').disabled = true;
        document.getElementById('posButton').disabled = true;
        
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
          <li><strong>A</strong> or <strong>←</strong>: Negative annotation</li>
          <li><strong>D</strong> or <strong>→</strong>: Positive annotation</li>
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
        <p>Predicted Probability: {prob:.3f}</p>
        {counter_html}
        {progress_html}
        {status_message}
        <img src="{img_base64}" width="512" height="512"/>
        {prefetch_html}
        <form method="POST" action="/label" id="labelForm">
          <input type="hidden" name="idx" value="{idx}"/>
          <input type="hidden" name="request_id" value="{request_id}"/>
          <input type="hidden" name="label" id="labelValue" value=""/>
          <button type="button" id="negButton" onclick="document.getElementById('labelValue').value='0';this.form.submit()">Negative (a/left)</button>
          <button type="button" id="posButton" onclick="document.getElementById('labelValue').value='1';this.form.submit()">Positive (d/right)</button>
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
        print(f"Error in prefetch: {e}")
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
        if label not in (0, 1):
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
