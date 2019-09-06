# ----------------------------------------------------------------------------------------------------------------------
import numpy
import os
import argparse
import detector_YOLO3
import tools_mAP
import tools_IO
import tools_YOLO
import tools_animation
# ----------------------------------------------------------------------------------------------------------------------
default_command     = 'E2E'

# ----------------------------------------------------------------------------------------------------------------------
folder_annotation = './data/ex_detector/bottles1/'
file_annotations  = './data/ex_detector/bottles1/markup.txt'
model_in          = './data/ex_detector/bottles1/model_default.h5'
metadata_in       = './data/ex_detector/bottles1/metadata_default.txt'
# ----------------------------------------------------------------------------------------------------------------------
folder_out  = './data/output/'
# ----------------------------------------------------------------------------------------------------------------------
def E2E(model_in,metadata_in,folder_annotation,file_annotations,full_folder_out,limit=10):

    file_annotations_train = full_folder_out + 'markup_train_true.txt'
    file_annotations_test = full_folder_out + 'markup_test_true.txt'
    tools_IO.split_annotation_file(folder_annotation, file_annotations, file_annotations_train, file_annotations_test,ratio=0.3, limit=limit)

    D = detector_YOLO3.detector_YOLO3(model_in, metadata_in,obj_threshold=0.01)

    file_markup_train_true = full_folder_out + 'markup_train_true.txt'
    file_markup_train_pred = full_folder_out + 'markup_train_pred.txt'
    file_markup_test_true  = full_folder_out + 'markup_test_true.txt'
    file_markup_test_pred  = full_folder_out + 'markup_test_pred.txt'


    D.learn(file_annotations_train, full_folder_out, folder_annotation,limit=limit)

    D.process_annotation(file_annotations_train,file_markup_train_true ,file_markup_train_pred, folder_annotation=folder_annotation,markup_only=True,limit=limit)
    D.process_annotation(file_annotations_test ,file_markup_test_true  ,file_markup_test_pred , folder_annotation=folder_annotation,markup_only=True,limit=limit)
    tools_mAP.plot_mAP_overlap(folder_annotation, file_markup_train_true, file_markup_train_pred, metadata_in, full_folder_out, out_prefix='train_')
    tools_mAP.plot_mAP_overlap(folder_annotation, file_markup_test_true, file_markup_test_pred, metadata_in, full_folder_out, out_prefix='test_')
    mAP_train = tools_mAP.plot_mAP_iou(folder_annotation, file_markup_train_true, file_markup_train_pred, metadata_in, full_folder_out, out_prefix='train_')
    mAP_test  = tools_mAP.plot_mAP_iou(folder_annotation, file_markup_test_true, file_markup_test_pred, metadata_in, full_folder_out, out_prefix='test_')

    log = [['data_source','num_last_layers','time_train', 'mAP_train', 'mAP_test'],[D.logs.data_source,D.logs.last_layers, D.logs.time_train, mAP_train, mAP_test]]
    tools_IO.save_mat(log, full_folder_out + 'log.txt')

    return
# ----------------------------------------------------------------------------------------------------------------------
def process_folder(model_in,metadata_in,folder_in,folder_out,limit):
    D = detector_YOLO3.detector_YOLO3(model_in, metadata_in, obj_threshold=0.1)
    D.process_folder(folder_in, folder_out,limit=limit)
    return
# ----------------------------------------------------------------------------------------------------------------------
def process_folder_negatives(model_in,metadata_in,folder_in,folder_out,limit,confidence=0.1):
    D = detector_YOLO3.detector_YOLO3(model_in, metadata_in, obj_threshold=confidence)
    D.process_folder_negatives(folder_in, folder_out,limit=limit,confidence=confidence)
    return
# ----------------------------------------------------------------------------------------------------------------------
def draw_boxes(folder_in,folder_annotation,folder_out):
    tools_mAP.draw_boxes(0, folder_annotation,folder_in+ 'markup_test_true.txt',folder_in + 'markup_test_pred.txt',folder_out, delim=' ', metric='recall', iou_th=None,ovp_th=0.60,ovd_th=0.50)
    return
# ----------------------------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='detector')

    parser.add_argument('--command',default=default_command)
    parser.add_argument('--model_in',default=model_in)
    parser.add_argument('--metadata_in',default=metadata_in)
    parser.add_argument('--folder_out',default=folder_out)
    parser.add_argument('--folder_annotation',default=folder_annotation)
    parser.add_argument('--file_annotations',default=file_annotations)
    parser.add_argument('--limit', default=100000,type=numpy.int)

    parser.add_argument('--folder_in', default=folder_annotation)

    args = parser.parse_args()

    full_folder_out = tools_IO.get_next_folder_out(args.folder_out)
    os.mkdir(full_folder_out)

    if args.command=='E2E':
        E2E(args.model_in,args.metadata_in,args.folder_annotation,args.file_annotations,full_folder_out,args.limit)

    if args.command=='process_folder':
        process_folder(args.model_in,args.metadata_in,args.folder_in,full_folder_out,args.limit)

    if args.command=='process_folder_negatives':
        process_folder_negatives(args.model_in,args.metadata_in,args.folder_in,full_folder_out,args.limit,confidence=0.1)

    if args.command=='draw_boxes':
        draw_boxes(args.folder_in,args.folder_annotation,full_folder_out)

    return
# ----------------------------------------------------------------------------------------------------------------------
    #tools_YOLO.draw_annotation_boxes('./data/ex_detector/LON/markup.txt', './data/ex_detector/LON/classes.txt', './data/ex_YOLO/models/metadata_default.txt', default_folder_out,delim=' ')
    #tools_mAP.plot_mAP_iou('./data/ex_detector/LON/', './data/ex_detector/LON/markup_test_true.txt','./data/ex_detector/LON/markup_test_pred.txt', './data/ex_YOLO/models/metadata_default.txt', default_folder_out,out_prefix='')
    #tools_mAP.plot_mAP_overlap('./data/ex_detector/LON/', './data/ex_detector/LON/markup_test_true.txt','./data/ex_detector/LON/markup_test_pred.txt', './data/ex_YOLO/models/metadata_default.txt', default_folder_out,out_prefix='')
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':


    E2E(model_in, metadata_in, folder_annotation, file_annotations, folder_out, limit=20)
    #tools_mAP.plot_mAP_iou('./data/ex_detector/bottles1/', './data/ex_detector/bottles1/markup_test_true.txt','./data/ex_detector/bottles1/markup_test_pred.txt', './data/ex_detector/bottles1/A_metadata.txt',folder_out, out_prefix='')
    #tools_mAP.draw_boxes(4, folder_annotation, folder_annotation+'markup_test_true.txt', folder_annotation+'markup_test_pred.txt',folder_out, delim=' ', metric='recall', iou_th=0.5, ovp_th=None, ovd_th=None)