# Py COCO Segmentor

A python utlitiy wrapper around `pycocotools` to generate a dataset for semantic segmentation from the original [COCO dataset](https://cocodataset.org/). This will generate a dataset consisting of a copy of images from COCO and masked images in the form of tiff files ready training on machine learning segmentation models like UNet.

## Dependencies

* `opencv-python`
* `numpy`
* `pycocotools`
* `fiftyone`
* annotation file from coco dataset
* images from coco dataset

(see usage on how to easily download images using [fiftyone](https://voxel51.com/docs/fiftyone/))

## Installation

Install dependencies:

```
pip install opencv-python numpy pycocotools fiftyone
```

## Example Usage

Download the dataset using `fiftyone`:

```
import fiftyone

dataset = fiftyone.zoo.load_zoo_dataset("coco-2017", label_types=["segmentations"], classes=["person", "bicycle", "car"], split="train")
```

This will create a directory called `fiftyone` in your home directory where the annotation files and raw images can be found in.

Create a directory `original` where the raw images will be dumped (copied) and `masks` where the generated tiff files will be saved.

```
mkdir original
mkdir masks
```

Generating a dataset using category `1` (person), `2` (bicycle) and `3` (car).

```
python -m pycocosegmentor  --annotation-file /path/to/annotations.json --image-dir /path/to/raw/images --original-img-dir original --mask-image-dir masks --category-ids 1 2 3
```
