
# Binary Annotation of Images with Active Learning
It's fast, it's informative, it's fun. 
What for? Some examples:
- annotate correct/incorrect image to later develop an error-checking model
- annotate correct/incorrect box to filter detections of a general detection model (like SAM) and develop a custom object detector

The bottleneck in annotation is human will. Just start annotating now and do it for a few hours, it is one of the most useful things to enhance performance you can do. 

## Active Learning
To annotate **good** and **quickly** we need to annotate the most informative samples. 
We propose two methods to choose the most informative samples to be annotated:
- entropy maximization: a classical active learning algorithm that works pretty well
- maximum probability: if you have few positive examples, it might be useful to annotate them all (same with negative, simply swap your class labels). Every correction is very valuable.

## Variations

- Image classification: say yes/no for each image
- Object detection: say yes/no for each bounding box

## Requirements
You just need:
- a list of (img_path,) OR
- a list of (img_path, bounding_box) 

just like in the `instance_references_EXAMPLE.txt`.

and of course, clone the repo and `pip install -r requirements.txt`.

### Compute features

The features are really important. We recommend using DINOv2reg or SigLIP2. To compute them run one of the following:
- `python compute_features.py --refs='instance_references_EXAMPLE.txt' --img --dino`
- `python compute_features.py --refs='instance_references_EXAMPLE.txt'--img --siglip`
- `python compute_features.py --refs='instance_references_EXAMPLE.txt'--bbox --dino`
- `python compute_features.py --refs='instance_references_EXAMPLE.txt'--bbox --siglip`

### Annotate
You can configure the `load_image` function in `app.py`, but it should work out of the box. To launch the webapp run: 

`uvicorn app:app --reload`

and go to `http://127.0.0.1:8000`.

#### Keyboard shortcuts:
- Press A / Left Arrow → Label as Negative.
- Press D / Right Arrow → Label as Positive.
- Press Backspace → Revert last annotation.









