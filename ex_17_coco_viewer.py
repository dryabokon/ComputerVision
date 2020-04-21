import numpy
import os
import cv2
from pycocotools.coco import COCO
# ----------------------------------------------------------------------------------------------------------------------
import tools_IO
import tools_YOLO
import tools_image
# ----------------------------------------------------------------------------------------------------------------------
def draw_annotation(filaname_coco_annnotation,folder_images,folder_out,draw_binary_masks=False):
    coco = COCO(filaname_coco_annnotation)
    colors = tools_IO.get_colors(1 + len(coco.cats))
    class_names = [coco.cats[key]['name'] for key in coco.cats]

    for key in coco.imgToAnns.keys():

        annotations = coco.imgToAnns[key]
        image_id = annotations[0]['image_id']
        filename = coco.imgs[image_id]['file_name']
        if not os.path.isfile(folder_images + filename):
            continue

        boxes, category_IDs = [], []
        for annotation in annotations:
            bbox = annotation['bbox']
            boxes.append([bbox[1], bbox[0], bbox[1] + bbox[3], bbox[0] + bbox[2]])
            category_IDs.append(annotation['category_id'])

        if draw_binary_masks:
            image = cv2.imread(folder_images + filename)
            result  = numpy.zeros_like(image)
            for box in boxes:
                top, left, bottom, right = box
                cv2.rectangle(result, (left, top), (right, bottom), (255,255,255), thickness=-1)
                cv2.imwrite(folder_out + filename.split('.')[0]+'.jpg', image)
        else:
            image = tools_image.desaturate(cv2.imread(folder_images + filename), level=0.8)
            result = tools_YOLO.draw_objects_on_image(image, boxes, [1] * len(boxes),category_IDs, colors, class_names)

        cv2.imwrite(folder_out + filename, result)
    return
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './data/output/'
folder_images = './data/ex_coco_view/'
filaname_coco_annnotation = './data/ex_coco_view/soccer_coco.json'
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    draw_annotation(filaname_coco_annnotation,folder_images,folder_out,draw_binary_masks=True)