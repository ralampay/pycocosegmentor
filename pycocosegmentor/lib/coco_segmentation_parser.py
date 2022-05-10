import json
import glob
from pycocotools.coco import COCO
import cv2
import numpy as np

class CocoSegmentationParser:
    def __init__(self, annotations_file, image_dir, category_ids=[]):
        self.annotations_file = annotations_file
        self.image_dir = image_dir
        self.coco = COCO(self.annotations_file)

        if len(category_ids) == 0:
            self.category_ids = self.get_all_category_ids()
        else:
            self.set_category_ids(category_ids)

    def set_category_ids(self, category_ids):
        self.category_ids = category_ids

    def get_all_category_ids(self):
        return self.coco.getCatIds()

    def get_all_categories(self):
        return self.coco.loadCats(self.get_all_category_ids())

    def get_categories(self):
        return self.coco.loadCats(self.category_ids)

    def build_dataset(self, original_image_dir, mask_image_dir):
        image_ids = self.coco.getImgIds(catIds=self.category_ids)

        images = self.coco.loadImgs(image_ids)

        for image_obj in images:
            image_file_name = image_obj['file_name']
            image_id        = image_obj['id']

            image_file_path = "{}/{}".format(self.image_dir, image_file_name)
            print("Processing {}...".format(image_file_path))

            image = cv2.imread(image_file_path)
            rows, cols, _ = image.shape

            image_file_to_write = "{}/{}".format(original_image_dir, image_file_name)
            print("Saving original image to {}...".format(image_file_to_write))
            cv2.imwrite(image_file_to_write, image)

            # Create empty image
            mask_image = np.zeros((rows, cols, 1), dtype=np.uint8)

            print("Processing mask for {}...".format(image_file_path))

            for i in range(len(self.category_ids)):
                category_id = self.category_ids[i]

                annotation_ids = self.coco.getAnnIds(imgIds=[image_id], catIds=[category_id])

                annotations = self.coco.loadAnns(annotation_ids)

                if len(annotations) > 0:
                    segs = []

                    for ann in annotations:
                        if(type(ann['segmentation']) == list):
                            segs.append(ann['segmentation'])

                    polygons = []

                    for seg in segs:
                        for s in seg:
                            polygons.append(np.array([s], np.int32).reshape((1, -1, 2)))

                    cv2.fillPoly(mask_image, polygons, color=[i+1])

            mask_image_file = "{}/{}".format(mask_image_dir, image_file_name).replace(".jpg", ".tiff")
            print("Writing mask file to {}...".format(mask_image_file))
            cv2.imwrite(mask_image_file, mask_image)
