from pathlib import Path
import pickle
from collections import defaultdict
import numpy as np

import matplotlib.pyplot as plt


def find_repeated_bboxes_and_feature_diffs(bboxes, feats):
    # Convert bboxes to tuples for hashing
    bbox_tuples = [tuple(b) for b in bboxes]

    # Find all indices for each bbox
    bbox_to_indices = defaultdict(list)
    for idx, bbox in enumerate(bbox_tuples):
        bbox_to_indices[bbox].append(idx)

    # Find repeated bboxes
    repeated = {bbox: idxs for bbox, idxs in bbox_to_indices.items() if len(idxs) > 1}
    print(f"Total repeated bounding boxes: {len(repeated)}")

    # Optionally, plot histogram of repetition counts
    repetition_counts = [
        len(idxs) for idxs in bbox_to_indices.values() if len(idxs) > 1
    ]
    if repetition_counts:
        plt.figure()
        plt.hist(repetition_counts, bins=range(2, max(repetition_counts) + 2))
        plt.xlabel("Number of repetitions")
        plt.ylabel("Number of bounding boxes")
        plt.title("Histogram of repeated bounding boxes")
        plt.savefig("repeated_bboxes_histogram.png")

    # Check for feature differences among repetitions
    bboxes_with_diff_feats = []
    for bbox, idxs in repeated.items():
        feat_set = {tuple(feats[i]) for i in idxs}
        if len(feat_set) > 1:
            bboxes_with_diff_feats.append((bbox, idxs))
            print(
                f"BBox {bbox} has {len(idxs)} repetitions with DIFFERENT features at indices {idxs}"
            )

    print(f"Repeated bboxes with different features: {len(bboxes_with_diff_feats)}")
    return repeated, bboxes_with_diff_feats


def nms_fast(bboxes, iou_threshold=0.95, distance_threshold=256):
    N = len(bboxes)
    used = np.zeros(N, dtype=bool)
    groups = []
    sort_idx = np.lexsort(
        (bboxes[:, 4], bboxes[:, 3], bboxes[:, 2], bboxes[:, 1], bboxes[:, 0])
    )
    sbboxes = bboxes[sort_idx]
    for i in range(N):
        print(f"Processing {i}/{N}, {i/N*100:.2f}%", end="\r")
        if used[i]:
            continue
        group = [sort_idx[i]]
        used[i] = True
        box = sbboxes[i]
        for j in range(i + 1, N):
            if used[j]:
                continue
            other = sbboxes[j]
            if other[0] != box[0]:
                break
            if other[1] - box[1] > distance_threshold:
                break
            if other[2] - box[2] > distance_threshold:
                continue
            iou = compute_iou(box[1:], other[1:])
            if iou > iou_threshold:
                group.append(sort_idx[j])
                used[j] = True
        if len(group) > 1:
            groups.append(group)
    return groups


def compute_iou(box1, box2):
    """
    box1, box2: [row, col, h, w]
    Returns IoU
    """
    y1 = max(box1[0], box2[0])
    x1 = max(box1[1], box2[1])
    y2 = min(box1[0] + box1[2], box2[0] + box2[2])
    x2 = min(box1[1] + box1[3], box2[1] + box2[3])

    inter_h = max(0, y2 - y1)
    inter_w = max(0, x2 - x1)
    inter = inter_h * inter_w

    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - inter

    return inter / (union + 1e-8)


def main():
    # we need these files
    REFS_FILE = "instance_references.txt"
    FEATURES_FILE = "features.npy"
    ANNOTATIONS_FILE = "cell_annotations.csv"

    # load cell dataset
    root = Path("/home/franchesoni/walden")
    with open(root / "cell_dataset_with_img4.pkl", "rb") as f:
        ds = pickle.load(f)
    print(ds["feats"].shape)
    # save features
    np.save(root / "image-active-learning" / FEATURES_FILE, ds["feats"])
    # save instances
    bboxes = ds["bboxes"].astype(int)
    np.savetxt(
        root / "image-active-learning" / REFS_FILE, bboxes, fmt="%d", delimiter=" "
    )
    # groups = nms_fast(bboxes, iou_threshold=0.95, distance_threshold=512)

    # annotations
    y_labeled = np.load(root / "walden-data" / "y_labeled.npy")
    labeled_bboxes = np.load(root / "walden-data" / "labeled_bboxes.npy")
    # look for the bboxes in the dataset, and find the indices of the labels, then save (index, label, image, bbox)
    # Find indices of labeled_bboxes in ds['bboxes']
    indices = []
    labels = []
    for bbox_ind, bbox in enumerate(labeled_bboxes):
        # Find all rows where all columns match
        matches = np.all(bboxes == bbox, axis=1)
        idx = np.where(matches)[0]
        if len(idx) == 0:
            print(f"Warning: bbox {bbox} not found in dataset!")
        elif len(idx) > 1:
            print(f"Warning: bbox {bbox} found multiple times!")
        for i in idx:
            indices.append(i)
            labels.append(y_labeled[bbox_ind])
    indices = np.array(indices)
    labels = np.array(labels)

    import csv

    for ind in range(len(indices)):
        index, label = indices[ind], labels[ind]
        with open(ANNOTATIONS_FILE, "a", newline="") as f:
            csv.writer(f).writerow([index, label] + list(bboxes[index]))
    print("Done preparing data!")


if __name__ == "__main__":
    main()
