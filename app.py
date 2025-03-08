import pickle
import tqdm
import h5py
import csv
import heapq
import io
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.optim as optim

from starlette.applications import Starlette
from starlette.responses import HTMLResponse, RedirectResponse, Response
from starlette.routing import Route
from starlette.staticfiles import StaticFiles

DATASET = ["cell", "full"][0]
COST = ["certainty", "minprob"][1]
CSVFILE = [
    "annotations_lymphocyte.csv",
    "annotations_is_cell.csv",
    "annotations_lymphoplasmocyte.csv",
    "annotations_plasmocyte.csv",
][3]


class DataManager:
    def __init__(self, root: Path):
        self.root = root
        self.labeled_indices = set()

        # Load features and bounding boxes
        print("Loading features and bounding boxes...")

        data = pickle.load(open(f"{DATASET}_dataset.pkl", "rb"))
        self.feats, self.bboxes = data["feats"], data["bboxes"]

        self.N = self.feats.shape[0]
        self.unlabeled_indices = set(range(self.N))

    def _load_bounding_boxes_csv(self, csv_file, imgnumber):
        bboxes = []
        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                bboxes.append([imgnumber] + [int(float(x)) for x in row])
        return np.array(bboxes)

    def get_features(self, idx):
        return self.feats[idx]

    def get_image(self, idx, center_crop=False):
        # Fetch the bounding box and load the corresponding image
        bbox = self.bboxes[idx]
        img = self._load_image_with_bbox(bbox, center_crop=center_crop)
        return img

    def _load_image_with_bbox(self, bbox, center_crop=False):
        """
        Load a 3x3 composite of tiles around the given bounding box and optionally
        return a 512x512 crop centered on the bounding box.

        Args:
            bbox (list or tuple): [imgnumber, row, col, height, width]
                - imgnumber: The image number (e.g., 5 for 'img5')
                - row, col: Top-left coordinates of the bounding box in global coordinates
                - height, width: Size of the bounding box
            center_crop (bool): If True, return a 512x512 crop centered on the bounding box.
                                If False, return the full 768x768 (3x3 tiles) composite.

        Returns:
            PIL.Image: The assembled image with bounding box drawn.
        """
        # Unpack the bounding box
        imgnumber, bbox_row, bbox_col, bbox_height, bbox_width = bbox

        TILE_SIZE = 256
        COMPOSITE_SIZE = TILE_SIZE * 3  # 768x768
        CROP_SIZE = 512
        HALF_CROP = CROP_SIZE // 2

        # Directory with tiles
        imgname = f"img{imgnumber}"
        tiles_dir = Path(self.root) / f"dataset/{imgname}/tiles"

        # Determine the tile grid start
        # We find the tile that contains the top-left corner of the bbox
        tile_row_start = (bbox_row // TILE_SIZE) * TILE_SIZE
        tile_col_start = (bbox_col // TILE_SIZE) * TILE_SIZE

        # Create a composite image (3x3 tiles)
        composite_image = Image.new(
            "RGB", (COMPOSITE_SIZE, COMPOSITE_SIZE), (255, 255, 255)
        )

        # Load surrounding 3x3 tiles
        for i, drow in enumerate([-TILE_SIZE, 0, TILE_SIZE]):
            for j, dcol in enumerate([-TILE_SIZE, 0, TILE_SIZE]):
                tile_row = tile_row_start + drow
                tile_col = tile_col_start + dcol
                tile_name = f"tile_{int(tile_row)}_{int(tile_col)}.jpeg"
                tile_path = tiles_dir / tile_name

                if tile_path.exists():
                    try:
                        tile_image = Image.open(tile_path)
                        if tile_image.mode != "RGB":
                            tile_image = tile_image.convert("RGB")
                        composite_image.paste(
                            tile_image, (j * TILE_SIZE, i * TILE_SIZE)
                        )
                    except Exception as e:
                        # If tile loading fails, use a placeholder
                        placeholder = Image.new(
                            "RGB", (TILE_SIZE, TILE_SIZE), (200, 200, 200)
                        )
                        draw_placeholder = ImageDraw.Draw(placeholder)
                        draw_placeholder.line(
                            (0, 0) + placeholder.size, fill=(150, 150, 150), width=3
                        )
                        draw_placeholder.line(
                            (0, placeholder.size[1], placeholder.size[0], 0),
                            fill=(150, 150, 150),
                            width=3,
                        )
                        composite_image.paste(
                            placeholder, (j * TILE_SIZE, i * TILE_SIZE)
                        )
                else:
                    # Missing tile placeholder
                    placeholder = Image.new(
                        "RGB", (TILE_SIZE, TILE_SIZE), (200, 200, 200)
                    )
                    draw_placeholder = ImageDraw.Draw(placeholder)
                    draw_placeholder.line(
                        (0, 0) + placeholder.size, fill=(150, 150, 150), width=3
                    )
                    draw_placeholder.line(
                        (0, placeholder.size[1], placeholder.size[0], 0),
                        fill=(150, 150, 150),
                        width=3,
                    )
                    composite_image.paste(placeholder, (j * TILE_SIZE, i * TILE_SIZE))

        # Draw the bounding box on the composite image
        # Calculate the bounding box coordinates relative to the composite image
        # The composite image's center tile corresponds to (tile_row_start, tile_col_start) in global coords
        # Top-left tile in composite is at (tile_row_start - TILE_SIZE, tile_col_start - TILE_SIZE)
        composite_top_row = tile_row_start - TILE_SIZE
        composite_left_col = tile_col_start - TILE_SIZE

        bbox_row_rel = bbox_row - composite_top_row
        bbox_col_rel = bbox_col - composite_left_col
        bbox_bottom_rel = bbox_row_rel + bbox_height
        bbox_right_rel = bbox_col_rel + bbox_width

        draw = ImageDraw.Draw(composite_image)
        draw.rectangle(
            [bbox_col_rel-10, bbox_row_rel-10, bbox_right_rel+10, bbox_bottom_rel+10],
            outline="green",
            width=3,
        )

        if center_crop:
            # We want to produce a 512x512 crop centered on the bbox center
            bbox_center_row = bbox_row_rel + bbox_height / 2
            bbox_center_col = bbox_col_rel + bbox_width / 2

            # Center the BBox in the crop
            # The BBox center should map to the center of the crop (256, 256)
            left = int(bbox_center_col - HALF_CROP)
            upper = int(bbox_center_row - HALF_CROP)
            right = left + CROP_SIZE
            lower = upper + CROP_SIZE

            # Ensure we don't go outside the composite image boundaries
            if left < 0:
                right -= left
                left = 0
            if upper < 0:
                lower -= upper
                upper = 0
            if right > COMPOSITE_SIZE:
                left -= right - COMPOSITE_SIZE
                right = COMPOSITE_SIZE
            if lower > COMPOSITE_SIZE:
                upper -= lower - COMPOSITE_SIZE
                lower = COMPOSITE_SIZE

            # Crop the image
            cropped_image = composite_image.crop((left, upper, right, lower))

            # If needed, pad to ensure exactly 512x512
            w, h = cropped_image.size
            if w < CROP_SIZE or h < CROP_SIZE:
                padded = Image.new("RGB", (CROP_SIZE, CROP_SIZE), (255, 255, 255))
                padded.paste(
                    cropped_image, ((CROP_SIZE - w) // 2, (CROP_SIZE - h) // 2)
                )
                cropped_image = padded

            return cropped_image
        else:
            return composite_image

    def mark_labeled(self, idx):
        self.labeled_indices.add(idx)
        self.unlabeled_indices.discard(idx)

    def unmark_labeled(self, idx):
        # Used during revert
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
    def __init__(self, n_features, n_iter=10):
        self.n_features = n_features
        self.n_iter = n_iter
        self.model = PyTorchLogisticRegression(self.n_features)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, weight_decay=0.01)
        self.loss_fn = nn.BCELoss()
        self.annotated_feats = []
        self.annotated_labels = []

    def add_annotation(self, feat, label):
        self.annotated_feats.append(feat)
        self.annotated_labels.append(int(label))

    def set_annotations(self, feats, labels):
        self.annotated_feats = feats
        self.annotated_labels = labels

    def fit(self):
        if len(self.annotated_feats) == 0:
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

    def predict_proba(self, X):
        if len(self.annotated_feats) == 0:
            # no training done, return 0.5
            return np.ones((len(X), 2)) * 0.5

        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            preds = self.model(X_t).numpy().flatten()
        # preds is probability of class 1
        probs = np.vstack([1 - preds, preds]).T
        return probs


class Orchestrator:
    def __init__(
        self,
        root_dir,
        annotation_file,
        score_interval=8,
        n_iter=10,
        sample_cost="certainty",
    ):
        assert sample_cost in ["certainty", "minprob"]
        self.sample_cost = sample_cost
        self.data_mgr = DataManager(root_dir)
        self.model = IncrementalModel(
            n_features=self.data_mgr.feats.shape[1], n_iter=n_iter
        )
        self.annotation_file = annotation_file
        self.annotations = []  # store (idx, label)

        self.current_sample = None
        self.previous_samples = []
        self.positive_count = 0
        self.negative_count = 0

        self.score_interval = score_interval
        self.candidates = np.random.choice(
            self.data_mgr.N, self.score_interval, replace=False
        ).tolist()  # get score_interval random samples initially
        self.current_sample = self.candidates.pop(0)
        self.next_sample = None  # For prefetching

        # If annotation file exists, load it and retrain to restore state
        if self.annotation_file.exists():
            self._load_annotations_and_retrain()

        self._ensure_next_sample()

    def _load_annotations_and_retrain(self):
        # Load existing annotations from file
        loaded_annotations = []
        with open(self.annotation_file, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                idx, label = int(row[0]), int(row[1])
                loaded_annotations.append((idx, label))
        # Retrain model
        self.retrain(loaded_annotations)

    def retrain(self, annotations):
        self.data_mgr.labeled_indices.clear()
        self.data_mgr.unlabeled_indices = set(range(self.data_mgr.N))
        for idx, _ in annotations:
            self.data_mgr.mark_labeled(idx)
        self.annotations = annotations.copy()
        feats = [self.data_mgr.get_features(idx) for (idx, _) in annotations]
        labels = [lbl for (_, lbl) in annotations]
        assert np.array([lbl in {0, 1} for lbl in labels]).all()
        self.model.set_annotations(feats, labels)
        self.model.fit()
        self.candidates.clear()

    def get_current_sample_index(self):
        return self.current_sample

    def get_next_sample_index(self):
        # Return next sample for prefetching if available
        return self.next_sample

    def handle_annotation(self, idx, label):
        # Save annotation
        with open(self.annotation_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([idx, label])

        # Update counters
        if label == 1:
            self.positive_count += 1
        elif label == 0:
            self.negative_count += 1

        # Update model
        feat = self.data_mgr.get_features(idx)
        self.model.add_annotation(feat, label)
        self.model.fit()
        self.data_mgr.mark_labeled(idx)
        self.annotations.append((idx, label))

        # Push current sample onto the previous_samples stack
        self.previous_samples.append(self.current_sample)

        # Move on to next sample
        self._pick_next_sample()

    def _pick_next_sample(self):
        if not self.candidates:
            self.refill_candidates()
        if self.candidates:
            self.current_sample = self.candidates.pop(0)
        else:
            self.current_sample = None

        self._ensure_next_sample()

    def _ensure_next_sample(self):
        # Prefetch next sample from candidates if available
        # If we have candidates, look at the next one in the heap
        if len(self.candidates) < 2:
            self.refill_candidates()
        if len(self.candidates) > 0:
            # Peek next candidate without popping it
            self.next_sample = self.candidates[0]
        else:
            self.next_sample = None

    def refill_candidates(self):
        unlabeled = self.data_mgr.get_unlabeled_indices()
        if not len(unlabeled):
            return
        X = self.data_mgr.feats
        probs = self.model.predict_proba(X)
        if self.sample_cost == "certainty":
            sample_cost = np.abs(probs[:, 1] - 0.5)
        elif self.sample_cost == "minprob":
            sample_cost = probs[:, 0]
        sorted_indices = np.argsort(sample_cost)
        self.candidates = [
            ind
            for ind in sorted_indices[: self.score_interval * 10].tolist()
            if ind in unlabeled
        ][: self.score_interval]

    def revert_last_annotation(self):
        if not self.annotations or not self.previous_samples:
            return

        # Pop last annotation and previous sample
        idx, label = self.annotations.pop()
        self.current_sample = self.previous_samples.pop()

        # Remove annotation from file: rewrite file
        with open(self.annotation_file, "w", newline="") as f:
            writer = csv.writer(f)
            for aidx, albl in self.annotations:
                writer.writerow([aidx, albl])

        # Unmark the last labeled sample
        self.data_mgr.unmark_labeled(idx)

        # Retrain model from scratch
        self.retrain(self.annotations)

    def get_probability_for(self, idx):
        # Return model probability for a single sample
        X = self.data_mgr.get_features(idx)[np.newaxis, :]
        probs = self.model.predict_proba(X)
        return probs[0, 1]


app_root = Path("")
orchestrator = Orchestrator(
    root_dir=app_root,
    annotation_file=Path(CSVFILE),
    score_interval=8,
    n_iter=10,
    sample_cost=COST,
)


async def homepage(request):
    idx = orchestrator.get_current_sample_index()
    if idx is None:
        return HTMLResponse("<h1>No more samples!</h1>")

    img = orchestrator.data_mgr.get_image(idx, center_crop=True)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()
    img_base64 = "data:image/jpeg;base64," + __import__("base64").b64encode(
        img_bytes
    ).decode("utf-8")

    # Probability
    prob = orchestrator.get_probability_for(idx)
    prob_html = f"<p>Predicted Probability (Positive): {prob:.3f}</p>"

    # Annotation Counters
    positive_count = orchestrator.positive_count
    negative_count = orchestrator.negative_count
    counter_html = f"""
    <p>Annotations: Positive = {positive_count}, Negative = {negative_count}</p>
    """

    # Prefetch next image if available
    next_idx = orchestrator.get_next_sample_index()
    prefetch_img_html = ""
    if next_idx is not None:
        # Prefetch route
        prefetch_img_html = f'<img id="prefetch" src="/prefetch?idx={next_idx}" style="display:none;" />'

    # JavaScript for key handling: 'a' or Left Arrow = Negative, 'd' or Right Arrow = Positive, Backspace = Revert
    script = """
    <script>
    document.addEventListener('keydown', function(event) {
        if (event.key === 'a' || event.key === 'ArrowLeft') {
            document.getElementById('negButton').click();
        } else if (event.key === 'd' || event.key === 'ArrowRight') {
            document.getElementById('posButton').click();
        } else if (event.key === 'Backspace') {
            event.preventDefault();
            window.location.href = '/revert';
        }
    });
    </script>
    """

    html = f"""
    <html>
    <body>
    <h1>Sample {idx}</h1>
    {prob_html}
    {counter_html}
    <img src="{img_base64}" width="512" height="512" />
    {prefetch_img_html}
    <form method="POST" action="/label" id="labelForm">
        <input type="hidden" name="idx" value="{idx}" />
        <button name="label" value="0" id="negButton">Negative (a/left)</button>
        <button name="label" value="1" id="posButton">Positive (d/right)</button>
    </form>
    {script}
    </body>
    </html>
    """
    return HTMLResponse(html)


async def prefetch_handler(request):
    idx = int(request.query_params["idx"])
    img = orchestrator.data_mgr.get_image(idx, center_crop=True)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return Response(buf.getvalue(), media_type="image/jpeg")


async def label_handler(request):
    form = await request.form()
    idx = int(form["idx"])
    label = int(form["label"])
    orchestrator.handle_annotation(idx, label)
    # Redirect immediately to homepage (no confirmation)
    return RedirectResponse("/", status_code=303)


async def revert_handler(request):
    orchestrator.revert_last_annotation()
    return RedirectResponse("/", status_code=303)


app = Starlette(
    debug=False,
    routes=[
        Route("/", homepage),
        Route("/label", label_handler, methods=["POST"]),
        Route("/prefetch", prefetch_handler),
        Route("/revert", revert_handler),
    ],
)
