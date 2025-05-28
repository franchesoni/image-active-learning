import tqdm
from pathlib import Path
import os
import numpy as np
import torch
import timm
from PIL import Image
import fire
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class InstancesDataset(Dataset):
    def __init__(self, refs_file, transform):
        self.transform = transform
        self.samples = []
        df = pd.read_csv(refs_file)
        for _, row in df.iterrows():
            img_path = row["filename"]
            bbox_str = row.get("bbox", "")
            if isinstance(bbox_str, str) and bbox_str.strip() != "":
                try:
                    # Expect bbox as 'top_row-left_col-bottom_row-right_col'
                    bbox = torch.tensor(list(map(int, bbox_str.strip().split("-"))))
                except Exception:
                    raise ValueError(f"Invalid bbox format in {img_path}: {bbox_str}")
            else:
                bbox = torch.tensor([-1, -1, -1, -1])
            self.samples.append({"img_path": img_path, "bbox": bbox})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample["img_path"]
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"{img_path} not found")
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)
        return img_tensor, sample["bbox"], img_path


def main(
    refs="instance_references_EXAMPLE.txt",
    out="features.npy",
    batch_size=16,
    num_workers=8,
    preserve=False,
):
    # Stick to DINO
    model_name = "vit_small_patch14_reg4_dinov2.lvd142m"
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.eval()
    model = model.cuda() if torch.cuda.is_available() else model

    # Get model-specific transform
    data_config = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**data_config, is_training=False)

    # Create dataset and dataloader
    dataset = InstancesDataset(refs, transform)
    dl = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    all_feats, all_paths = [], []
    # Assume patch size from model (default 14 for vit_large_patch14)
    patch_size = getattr(model, "patch_size", 14)
    input_size = data_config.get("input_size", (3, 224, 224))
    feat_n_rows, feat_n_cols = input_size[1] // patch_size, input_size[2] // patch_size

    tmppath = Path("tmp.npy")
    if tmppath.exists():
        os.remove(tmppath)
    with open(tmppath, "ab") as f:
        with torch.no_grad():
            for imgs, bboxes, paths in tqdm.tqdm(dl):
                imgs = imgs.cuda() if torch.cuda.is_available() else imgs
                feats_out = model.forward_features(imgs)
                B, Np5, D = feats_out.shape
                # Expecting dict with 'x_norm_clsfeat' and optionally 'x_norm_patchfeats'
                cls_feats = feats_out[:, 0, :]  # (B, D)
                patch_feats = feats_out[:, 5:].reshape(
                    B, feat_n_rows, feat_n_cols, D
                )  # (B, fH, fW, D)

                batch_feats = []
                for i in range(imgs.size(0)):
                    bbox = bboxes[i]
                    if (-1 == bbox).any():
                        feat = cls_feats[i]  # (D,)
                    else:
                        r1, c1, r2, c2 = bbox
                        # Map bbox coordinates (assumed in resized image space) to feat grid indices
                        feat_r1 = int(r1 / patch_size)
                        feat_c1 = int(c1 / patch_size)
                        feat_r2 = int(np.ceil(r2 / patch_size))
                        feat_c2 = int(np.ceil(c2 / patch_size))
                        feats = patch_feats[i]  # (fH, fW, D)
                        feat_r1 = max(feat_r1, 0)
                        feat_c1 = max(feat_c1, 0)
                        feat_r2 = min(feat_r2, feat_n_rows)
                        feat_c2 = min(feat_c2, feat_n_cols)
                        if feat_r2 <= feat_r1 or feat_c2 <= feat_c1:
                            feat = cls_feats[i]
                        else:
                            region = feats[feat_r1:feat_r2, feat_c1:feat_c2, :]
                            feat = region.mean(dim=(0, 1))
                    batch_feats.append(feat)
                    all_paths.append(paths[i])
                batch_feats = torch.stack(batch_feats).float().cpu().numpy()
                batch_feats.tofile(f)
    feats = np.fromfile(tmppath, dtype=np.float32).reshape(-1, D)
    np.save(out, feats)
    os.remove(tmppath)
    # update references with feature indices
    if not preserve:  # don't preserve the original csv, add to it
        df = pd.read_csv(refs)
        df["feat_idx"] = [i for i in range(len(all_paths))]
        df.to_csv(refs, index=False)


def convert_to_csv(
    path_to_filenames,
    out="refs.csv",
):
    with open(path_to_filenames, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines if line.strip()]
    df = {
        "filename": lines,
        "bbox": [""] * len(lines),
        "annotation": [""] * len(lines),
        "feat_idx": [""] * len(lines),
    }
    pd.DataFrame(df).to_csv(out, index=False)


if __name__ == "__main__":
    funcs = {"main": main, "csv": convert_to_csv}
    fire.Fire(funcs)
