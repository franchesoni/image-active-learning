import os
import numpy as np
import torch
import timm
from PIL import Image
import fire
from torch.utils.data import Dataset, DataLoader

class InstancesDataset(Dataset):
    def __init__(self, refs_file, transform):
        self.transform = transform
        self.samples = []
        with open(refs_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                img_path = parts[0]
                bbox = torch.tensor(list(map(int, parts[1:5]))) if len(parts) == 5 else torch.tensor([-1, -1, -1, -1])
                self.samples.append({'img_path': img_path, 'bbox': bbox})
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample['img_path']
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"{img_path} not found")
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        return img_tensor, sample['bbox'], img_path

def main(refs='instance_references_EXAMPLE.txt', out='features.npy', batch_size=16, num_workers=8):
    # Stick to DINO
    model_name = 'vit_small_patch14_reg4_dinov2.lvd142m'
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.eval()
    model = model.cuda() if torch.cuda.is_available() else model

    # Get model-specific transform
    data_config = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**data_config, is_training=False)

    # Create dataset and dataloader
    dataset = InstancesDataset(refs, transform)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    all_feats, all_paths = [], []
    # Assume patch size from model (default 14 for vit_large_patch14)
    patch_size = getattr(model, 'patch_size', 14)
    input_size = data_config.get('input_size', (3, 224, 224))
    grid_h, grid_w = input_size[1] // patch_size, input_size[2] // patch_size

    with torch.no_grad():
        for imgs, bboxes, paths in dl:
            imgs = imgs.cuda() if torch.cuda.is_available() else imgs
            feats_out = model.forward_features(imgs)
            # Expecting dict with 'x_norm_clstoken' and optionally 'x_norm_patchtokens'
            cls_tokens = feats_out.get('x_norm_clstoken', None) if isinstance(feats_out, dict) else feats_out
            patch_tokens = feats_out.get('x_norm_patchtokens', None) if isinstance(feats_out, dict) else None

            batch_feats = []
            for i in range(imgs.size(0)):
                bbox = bboxes[i]
                if bbox is None or patch_tokens is None:
                    feat = cls_tokens[i]
                else:
                    x1, y1, x2, y2 = bbox
                    # Map bbox coordinates (assumed in resized image space) to token grid indices
                    token_x1 = int(x1 / patch_size)
                    token_y1 = int(y1 / patch_size)
                    token_x2 = int(np.ceil(x2 / patch_size))
                    token_y2 = int(np.ceil(y2 / patch_size))
                    tokens = patch_tokens[i].reshape(grid_h, grid_w, -1)
                    token_x1 = max(token_x1, 0)
                    token_y1 = max(token_y1, 0)
                    token_x2 = min(token_x2, grid_w)
                    token_y2 = min(token_y2, grid_h)
                    if token_x2 <= token_x1 or token_y2 <= token_y1:
                        feat = cls_tokens[i]
                    else:
                        region = tokens[token_y1:token_y2, token_x1:token_x2, :]
                        feat = region.mean(dim=(0,1))
                batch_feats.append(feat.cpu().numpy())
                all_paths.append(paths[i])
            all_feats.append(np.stack(batch_feats))
    all_feats = np.concatenate(all_feats, axis=0) if all_feats else np.array([])
    np.save(out, all_feats)
    print(f"Saved features to {out}, shape={all_feats.shape}")

if __name__ == '__main__':
    fire.Fire(main)
