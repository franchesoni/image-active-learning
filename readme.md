# Annotate Images into Classes Fast! 

A fast, efficient tool for annotating images with labels using active learning to maximize annotation efficiency.

## Requirements

**What you need to try it:**
- a `.csv` file with columns `filename`, `bbox`, `annotation`, and `feat_idx`
- a `.npy` file with a feature vector associated to each image
- a prepared environment

### Need help getting those ready?
**Environment:**
- download `uv` or your preferred package manager: `curl -LsSf https://astral.sh/uv/install.sh | sh` (if you don't use `uv` the commands change a little)
- create your virtual environment: `uv venv --python=3.12` (other python versions weren't tested but should work) and activate it `source .venv/bin/activate`
- install the requirements `uv pip install -r requirements.txt`

**`.csv` file**
We have some example data. Initially it only contained the images. Then we run:
- `find example_data/images/ -type f -name "*.jpg" > example_data/filenames.txt` to create a list of images
- `python compute_features.py csv --path_to_filenames='example_data/filenames.txt' --out='example_data/refs.csv'` to convert the list of files to the `.csv` format the app expects

**`.npy` file**
You can compute the features yourself however you think best, but the DINOv2reg-small baseline is provided as an starting point:
- run `python compute_features.py main --refs='example_data/refs.csv' --out='example_data/features.npy'` to create the feature file and add the feature index for each sample to the `.csv`

If you do it yourself, remember that `.npy` should be a matrix and that each of the `feat_idx` values in the `.csv` should correspond to the row in the matrix that contains the feature vector for the sample. 

## Demo
Starting from your own folder with images, you can follow the steps above. If you want to see what the app looks like (and you have cloned the repo and set up your environment), run

`uvicorn app:app --port 8001`

to launch the app, and open [localhost:8001](http://127.0.0.1:8001) in your browser to start annotating. Note that the configuration settings for the app (such as the class names) are at the top of `app.py`.


## Pitch
This tool provides a web-based interface for image annotation with active learning strategies that prioritize the most informative samples, helping you:

- Build high-quality datasets quickly
- Focus human effort on the most valuable samples
- Monitor model improvement in real time

## Use Cases

- **Error identification**: Annotate correct/incorrect images to train error-checking models
- **Detection refinement**: Filter bounding box proposals from general detection models (like SAM)  
- **Object classification**: Quickly build binary classifiers (e.g., defective vs. non-defective parts)
- **Data cleaning**: Identify and remove outliers or mislabeled images
- **General annotation**: Simply create your own ImageNet dataset.

## Features

- **Optimized sample selection** using active learning (see below!)
- **Keyboard shortcuts** for rapid annotation
- **Real-time error metrics** to track model improvement. This is cool: every sample you annotate is an unseen sample for the model, therefore we can compute a generalization error while you annotate, and see if the model improves!
- **Progress tracking** with visual indicators
- **Annotation reverting** capability
- **Prefetching** for smoother experience

### Active Learning Strategy

We use an uncertainty-based method that also accounts for the number of annotations in each class. Mathematically, we compute the score $s$ for each sample as: 

$$ s = (1- p(c_\max)) / (A(c_\max)+1)  $$

where $c_\max = \arg\max_c p(c)$ is the probability of the predicted class and $A(c)$ is the number of annotations made so far for class $c$. 





### Annotation Controls
- A / Left Arrow: Label as Negative
- D / Right Arrow: Label as Positive
- Backspace: Revert last annotation
- H / ?: Show keyboard shortcuts

### Performance Tips
- Start with just 5 minutes of annotation to get early results
- Take short breaks to avoid annotation fatigue
- The model improves as you annotate more samples, so efficiency increases over time
- Expect about 1000 images per hour on average (around 3s per image).

### Advanced Configuration
Edit the constants at the top of `app.py` to customize paths and behavior:

### Interpreting the Error Chart
The error visualization shows:

- Red points: Individual mean squared errors (MSE) (lower is better)
- Red line: Average MSE 
- Blue line: Error rate (percentage of incorrect predictions)
For both, lower is better!


