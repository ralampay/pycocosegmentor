import sys
import argparse
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from lib.coco_segmentation_parser import CocoSegmentationParser

def main():
    parser = argparse.ArgumentParser(description="PyCocoSegmentor: Python utility for COCO segmentation dataset generation")

    parser.add_argument("--annotations-file", help="Location of the COCO annotation file", required=True)
    parser.add_argument("--image-dir", help="Location of downloaded COCO images", required=True)
    parser.add_argument("--category-ids", help="List of category ids to filter (will fetch all if none)", type=int, nargs='+', default=[])
    parser.add_argument("--original-image-dir", help="Location where original images will be copied to", type=str, required=True)
    parser.add_argument("--mask-image-dir", help="Location where tiff masks will be saved", type=str, required=True)

    args = parser.parse_args()

    annotations_file    = args.annotations_file
    image_dir           = args.image_dir
    category_ids        = args.category_ids
    original_image_dir  = args.original_image_dir
    mask_image_dir      = args.mask_image_dir

    coco_parser = CocoSegmentationParser(
        annotations_file=annotations_file,
        image_dir=image_dir,
        category_ids=category_ids
    )

    coco_parser.build_dataset(
        original_image_dir=original_image_dir,
        mask_image_dir=mask_image_dir
    )

if __name__ == '__main__':
    main()
