import csv
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
import base64

# Configs
REFS_FILE = "instance_references_EXAMPLE.txt"
FEATURES_FILE = "features.npy"
ANNOTATIONS_FILE = "annotations.csv"


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

    def get_features(self, idx):
        return self.feats[idx]

    def get_image(self, idx, center_crop=False):
        sample = self.samples[idx]
        img = Image.open(sample["img_path"]).convert("RGB")
        if sample["bbox"]:
            draw = ImageDraw.Draw(img)
            r1, c1, r2, c2 = sample["bbox"]
            draw.rectangle([c1, r1, c2, r2], outline="green", width=3)
            if center_crop:
                cm, rm = (c1 + c2) // 2, (r1 + r2) // 2
                left = max(cm - 256, 0)
                upper = max(rm - 256, 0)
                img = img.crop((left, upper, left + 512, upper + 512))
            else:
                img = img.resize((512, 512))
        else:
            img = img.resize((512, 512))
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
    def __init__(self, n_features, n_iter=10):
        self.n_features = n_features
        self.n_iter = n_iter
        self.model = PyTorchLogisticRegression(n_features)
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

    def predict_proba(self, X):
        if not self.annotated_feats:
            return np.ones((len(X), 2)) * 0.5
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            preds = self.model(X_t).numpy().flatten()
        return np.vstack([1 - preds, preds]).T


class Orchestrator:
    def __init__(
        self,
        refs_file,
        annotation_file,
        score_interval=8,
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
        self.candidates = np.random.choice(
            self.data_mgr.N, score_interval, replace=False
        ).tolist()
        self.current_sample = self.candidates.pop(0)
        self.next_sample = None
        if self.annotation_file.exists():
            self._load_annotations_and_retrain()
        self._ensure_next_sample()

    def _load_annotations_and_retrain(self):
        loaded = []
        with open(self.annotation_file, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                idx, label = int(row[0]), int(row[1])
                loaded.append((idx, label))
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

    def handle_annotation(self, idx, label):
        with open(self.annotation_file, "a", newline="") as f:
            csv.writer(f).writerow([idx, label])
        if label == 1:
            self.positive_count += 1
        else:
            self.negative_count += 1
        feat = self.data_mgr.get_features(idx)
        self.model.add_annotation(feat, label)
        self.model.fit()
        self.data_mgr.mark_labeled(idx)
        self.annotations.append((idx, label))
        self.previous_samples.append(self.current_sample)
        self._pick_next_sample()

    def _pick_next_sample(self):
        if not self.candidates:
            self.refill_candidates()
        self.current_sample = self.candidates.pop(0) if self.candidates else None
        self._ensure_next_sample()

    def _ensure_next_sample(self):
        if len(self.candidates) < 2:
            self.refill_candidates()
        self.next_sample = self.candidates[0] if self.candidates else None

    def refill_candidates(self):
        unlabeled = self.data_mgr.get_unlabeled_indices()
        if not len(unlabeled):
            return
        X = self.data_mgr.feats
        probs = self.model.predict_proba(X)
        cost = (
            np.abs(probs[:, 1] - 0.5)
            if self.sample_cost == "certainty"
            else probs[:, 0]
        )
        sorted_idx = np.argsort(cost)
        self.candidates = [
            i for i in sorted_idx[: self.score_interval * 10].tolist() if i in unlabeled
        ][: self.score_interval]

    def revert_last_annotation(self):
        if not self.annotations or not self.previous_samples:
            return
        idx, label = self.annotations.pop()
        self.current_sample = self.previous_samples.pop()
        with open(self.annotation_file, "w", newline="") as f:
            writer = csv.writer(f)
            for aidx, albl in self.annotations:
                writer.writerow([aidx, albl])
        self.data_mgr.unmark_labeled(idx)
        self.retrain(self.annotations)

    def get_probability_for(self, idx):
        X = self.data_mgr.get_features(idx)[np.newaxis, :]
        probs = self.model.predict_proba(X)
        return probs[0, 1]


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
        return HTMLResponse("<h1>No more samples!</h1>")
    img = orchestrator.data_mgr.get_image(idx, center_crop=True)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_base64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(
        "utf-8"
    )
    prob = orchestrator.get_probability_for(idx)
    counter_html = f"<p>Annotations: Positive={orchestrator.positive_count}, Negative={orchestrator.negative_count}</p>"
    next_idx = orchestrator.get_next_sample_index()
    prefetch_html = (
        f'<img id="prefetch" src="/prefetch?idx={next_idx}" style="display:none;" />'
        if next_idx is not None
        else ""
    )
    script = """
    <script>
    document.addEventListener('keydown', function(event) {
      if(event.key==='a' || event.key==='ArrowLeft') document.getElementById('negButton').click();
      else if(event.key==='d' || event.key==='ArrowRight') document.getElementById('posButton').click();
      else if(event.key==='Backspace'){ event.preventDefault(); window.location.href='/revert'; }
    });
    </script>
    """
    html = f"""
    <html>
      <body>
        <h1>Sample {idx}</h1>
        <p>Predicted Probability: {prob:.3f}</p>
        {counter_html}
        <img src="{img_base64}" width="512" height="512"/>
        {prefetch_html}
        <form method="POST" action="/label" id="labelForm">
          <input type="hidden" name="idx" value="{idx}"/>
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
