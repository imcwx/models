from os import walk
from os.path import join
import argparse
from copy import deepcopy

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.seterr(divide='ignore', invalid='ignore')


parser = argparse.ArgumentParser()
parser.add_argument("-pj", "--proj", default="test_images", 
                    type=str, help="proj folder name")
parser.add_argument("-tr", "--truth", default="ground_truth", 
                    type=str, help="ground truth folder name")
parser.add_argument("-pd", "--preds", default=["result"], nargs='+',
                    type=str, help="list of predictions annotation folder name")
parser.add_argument("-th", '--thres', default=[0.4], nargs='+', type=float, help="IOU thres")
parser.add_argument("-i", "--ignore", default=True,
                    type=bool, help="Ignore if true else include those boxes.")
args = parser.parse_args()


def batch_iou_new(boxes, box):
    """Compute the Intersection-Over-Union of a batch of boxes with another box.
    Args:
      box1: 2D array of [x, y, width, height].
      box2: a single array of [x, y, width, height]
    Returns:
      ious: array of a float number in range [0, 1].
    """
    if len(boxes) == 0:
        return 0
    lr = np.maximum(np.minimum(boxes[:,0]+boxes[:,2], box[0]+box[2]) - \
                    np.maximum(boxes[:,0], box[0]),
                    0)
    tb = np.maximum(np.minimum(boxes[:,1]+boxes[:,3], box[1]+box[3]) - \
                    np.maximum(boxes[:,1], box[1]),
                    0)
    inter = lr*tb
    union = boxes[:,2]*boxes[:,3] + box[2]*box[3] - inter
    return inter*1.0/union


def filenames(mypath):
    f = []
    for (dirpath, dirnames, filenames) in walk(mypath):
        for file in filenames:
            if not file.startswith('.'):
                f.append(file)
    return f


def check_files(truth, preds):
    f_gt = filenames(truth)
    f_pred = filenames(preds)
    # temp = [x.lower() for x in f_gt]
    # temp2 = [x.lower() for x in f_pred]
    # assert temp == temp2 , "Files are not the same!"
    assert f_gt == f_pred , "Files are not the same!"
    # tmp = ["IMG_7513.JPG.json"]
    # f_gt = tmp
    # f_pred = tmp
    return f_gt, f_pred


def get_mapping(template):
    with open(template, "r") as json_file:
        json_data = json.load(json_file)
    list_of_dict = json_data["categories"][0]["skus"]
    idx_to_name = {}
    name_to_idx = {}
    class_name = []
    for item in list_of_dict:
        idx_to_name[item["id"]] = item["name"]
        name_to_idx[item["name"]] = item["id"]
        class_name.append(item["name"])
    return class_name, idx_to_name, name_to_idx


def make_pd(class_name):
    n = len(class_name)
    column_names = class_name+['Not labelled']
    row_names = class_name+['Miss']
    df_con_mat = pd.DataFrame(np.zeros((n+1, n+1)), columns=column_names, index=row_names)
    return df_con_mat


def get_one(json_path):
    with open(json_path, "r") as json_file:
        json_data = json.load(json_file)
    idxs = []
    boxes = []
    for item in json_data["bndboxes"]:
        try:
            ignore = item['ignore']
        except KeyError as keyErr:
            # print("bnbbox {} key error in conflict annotations".format(keyErr))
            ignore = False

        if args.ignore and ignore:
            # black_bndboxes.append(bndbox)
            pass
        elif item['id'] == 'nil':
            # print("ignore id nil bnbbox")
            pass
        else:
            idxs.append(item["id"])
            boxes.append([item["x"], item["y"], item["w"], item["h"] ])
    return idxs, boxes


def compare_one_new(json_true, json_predict, list_of_df, idx_to_name, thres):
    gt_idxs, gt_boxes = get_one(json_true)
  #  print("the file", json_true)
  #  print("gt_idxs", gt_idxs)
  #  print("gt_boxes", gt_boxes)
    pred_idxs, pred_boxes = get_one(json_predict)
    gt = [(c , b) for c ,b in zip(gt_idxs, gt_boxes)]
    list_of_keep = []
    for _ in thres:
        list_of_keep.append(np.zeros(len(pred_idxs)))
    for (gt_idx, gt_box) in gt:
        overlaps = batch_iou_new(np.array(pred_boxes), gt_box)
        for i in range(len(thres)):
            keep = (np.array(overlaps) >= thres[i])
            list_of_keep[i] += keep
            if np.sum(keep) == 0: # if predicted nothing, therefore miss
                gt_name = idx_to_name[gt_idx]
                list_of_df[i][gt_name]["Miss"] += 1
            else:
                for idx in np.argwhere(keep)[0]:
                    pred_idx = pred_idxs[idx]
                    gt_name = idx_to_name[gt_idx]
                    pred_name = idx_to_name[pred_idx]
                    list_of_df[i][gt_name][pred_name] += 1 # add to matrix
                    # if gt_name != pred_name:
                    #     print("  gt_name: " + gt_name)
                    #     print("pred name: " + pred_name)
    for i in range(len(thres)): 
        for idx in np.argwhere(list_of_keep[i] == 0).T[0]: # Detection but has no label
            pred_idx = pred_idxs[idx]
            # pred_box = pred_boxes[idx]
            pred_name = idx_to_name[pred_idx]
            list_of_df[i]["Not labelled"][pred_name] += 1
            # print("Not labbelled: " + pred_name)


def eval():
    gt_dir = join(args.proj, args.truth)
    class_name, idx_to_name, name_to_idx = get_mapping(join(args.proj, 'templates.json'))  # once
    dfs = {}
    for preds in args.preds:
        pred_dir = join(args.proj, preds, "Annotations")
        gt_file, pred_file = check_files(gt_dir, pred_dir)
        df_con_mat = make_pd(class_name)

        list_of_df = []
        for i in args.thres:
            list_of_df.append(deepcopy(df_con_mat))

        for i in range(len(gt_file)):
            # print(" ")
            # print(gt_file[i])
            compare_one_new(join(gt_dir, gt_file[i]), 
                            join(pred_dir, pred_file[i]), 
                            list_of_df, idx_to_name, args.thres )
        dfs[preds] = list_of_df
        # for i in range(len(args.thres)):
        #     list_of_df[i].to_csv('{}/{}_{}_con_mat.csv'.format(join(args.proj) , preds, args.thres[i]),
        #                         sep=',', mode='w')
    return dfs


###################################################################### EVAL done, now compare ##################################################


def create_summary(class_names):
    column_names = class_names+["Miss", "Overall"]
    row_names=[]
    s='-'
    for pred in args.preds:
        for thres in args.thres:
            acc = s.join((pred,str(thres),'acc'))
            # prec = s.join((pred,str(thres),'prec'))
            row_names.extend([acc])
            # row_names.extend([acc,prec])
    num_col = len(column_names)
    num_row = len(row_names)
    summary = pd.DataFrame(np.zeros((num_row, num_col)), columns=column_names, index=row_names)
    return summary


def eval_con_mat(summary, df_con_mat, pred, thres):
    # Setup 
    s='-'
    acc = s.join((pred,str(thres),'acc'))
    # prec = s.join((pred,str(thres),'prec'))

    # Overall score
    diag = np.sum(np.diag(df_con_mat))
    total = df_con_mat.values.sum()
    TP = diag/total
    Miss = np.sum(df_con_mat.T["Miss"])/total
    summary["Overall"][acc] = TP
    summary["Miss"][acc] = Miss
    # print(toal)
    print("%25s:%f" %(pred+"-"+str(thres), TP))
    # print("%20s:%i" %("Miss", Miss))
    # print(pred)
    # print(TP)
    # print(total)

    class_name = list(summary.columns.values)[:-2]
    # Individual score
    for name in class_name:
        eac_sku_total_acc = np.sum(df_con_mat[name])
        eac_sku_acc = (df_con_mat[name][name]/eac_sku_total_acc)
        summary[name][acc] = eac_sku_acc

        # eac_sku_total_prec = np.sum(df_con_mat.loc[name])
        # eac_sku_prec = (df_con_mat[name][name]/eac_sku_total_prec)
        # summary[name][prec] = eac_sku_prec
    

def compare(dfs):
    start_once = True
    for pred in args.preds:
        list_of_df = dfs[pred]
        # for thres in args.thres:
        for i in range(len(args.thres)):
            thres = args.thres[i]
            df_con_mat = list_of_df[i]
            # df_con_mat = pd.DataFrame.from_csv('{}/{}_{}_con_mat.csv'.format(join(args.proj) , pred, thres))
            if start_once:
                class_names = list(df_con_mat.columns.values)[:-1]
                summary = create_summary(class_names)
                start_once = False
            eval_con_mat(summary, df_con_mat, pred, thres)
    
    print(" ")
    print("Mean Accuracy:")
    print(np.mean(summary.T[:-2]))

    summary.T.to_csv('summary.csv', sep=',', mode='w')
    
    TO_PLOT = False
    if TO_PLOT:
	    f = plt.figure()
	    summary.T.plot(kind='bar', ax=f.gca())
	    plt.legend(loc='best', bbox_to_anchor=(1.0, 0.5))
	    plt.tight_layout()
	    f.subplots_adjust(right=0.85)
	    plt.show()


if __name__ == "__main__":
    dfs = eval()
    compare(dfs)


