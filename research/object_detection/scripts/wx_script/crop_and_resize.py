import argparse
import cv2
import json
from shutil import copytree
import os
import fnmatch
from os.path import abspath, join, splitext, basename
import numpy as np
import time
import threading

parser = argparse.ArgumentParser()
parser.add_argument("--rm_ignore", default=True, help="Blackout ignore flags", type=bool)
parser.add_argument("--to_crop_sku", default=False, help="To Crop a boundary around the SKUs", type=bool)

parser.add_argument("--to_min_pixel", default=False, help="To crop to Minimum Pixel", type=bool)
parser.add_argument("--min_pixel", default=80*80, help="Minimum Pixel", type=int)

parser.add_argument("--to_resize_pad", default=False, help="To resize first than pad a boundary around the image", type=bool)
parser.add_argument("--to_crop_resize_pad", default=True, help="To crop, resize first than pad a boundary around the image", type=bool)

parser.add_argument("--target_h", default=int(800), type=int, help="target height")
parser.add_argument("--target_w", default=int(600), type=int, help="target width")

parser.add_argument("--input_folder", default="RB_Train_flag_2", type=str, help="put entire directory")
parser.add_argument("--out_folder", help=None, type=str)
args = parser.parse_args()

DEBUG = False


def find_image_files(out_folder):
    photos_path = join(out_folder, 'photos')
    if not os.path.exists(photos_path):
        photos_path = join(out_folder, 'Photos')
    assert os.path.exists(photos_path), "photo folder is not found at {}".format(photos_path)
    all_files = os.listdir(photos_path)
    photo_files = []
    for file in all_files:
        if fnmatch.fnmatch(file, '*.jpg') or fnmatch.fnmatch(file, '*.JPG'):
            photo_files.append(os.path.join(photos_path, file))
    return photo_files, photos_path


def find_annotation_files(photos_path):
    annotation_path= join(photos_path, "Annotations")
    assert os.path.exists(annotation_path), "Annotation folder is not found at {}".format(annotation_path)
    all_files = os.listdir(annotation_path)
    annotation_files = []
    for file in all_files:
        if fnmatch.fnmatch(file, '*.json') or fnmatch.fnmatch(file, '*.JSON'):
            annotation_files.append(os.path.join(annotation_path, file))
    return annotation_files


def find_files(out_folder):
    photo_files, photos_path = find_image_files(out_folder)
    annotation_files = find_annotation_files(photos_path)
    # assert len(photo_files) == len(annotation_files), "Number of images and annotations are not the same!"
    files = [(c, b) for c, b in zip(photo_files, annotation_files)]

    def check_files(photo_file, annotation_file):
        annot = splitext(basename(annotation_file))[0]
        photo = basename(photo_file)
        assert annot == photo, "Mismatch in annotations: {}.json and image: {}".format(annot, photo)

    for (photo_file, annotation_file) in files:
        check_files(photo_file, annotation_file)
    return files


def get_info(json_data, info):
    height = info["image_height"] = json_data["image_height"]
    width = info["image_width"] = json_data["image_width"]
    info["image_file"] = json_data["filename"]
    info["No_sku"] = False

    if len(json_data["bndboxes"]) == 0:  # no sku
        info["No_sku"] = True
        pass
    else:
        xmin, xmax, ymin, ymax, w, h, area = 10000, 0, 10000, 0, [], [], []  # arbitrary
        for bndboxes in json_data["bndboxes"]:
            xmin = min(xmin, bndboxes["x"])
            ymin = min(ymin, bndboxes["y"])
            xmax = max(xmax, (bndboxes["x"] + bndboxes["w"]))
            ymax = max(ymax, (bndboxes["y"] + bndboxes["h"]))
            if xmax > width or ymax > height:
                print("There is a bndbox outside of height and width")
                print("The filename is : " + json_data["filename"])
                print("xmax: {} , width : {}".format(xmax, width))
                print("ymax: {} , height : {}".format(ymax, height))
                print("The bndbox is : {} ".format(bndboxes))
            area.append(bndboxes["w"]*bndboxes["h"])
        info["xmin"], info["xmax"] = xmin, xmax
        info["ymin"], info["ymax"] = ymin, ymax
        info["area"] = sorted(area)
        info["crop_h"] = ymax - ymin
        info["crop_w"] = xmax - xmin


def get_crop_margin(info, fix_h=100, fix_w=100):
    fixed = True
    if info["No_sku"]:  # no sku
        pass
    else:
        if fixed:
            margin_h = fix_h
            margin_w = fix_w
        else:
            margin_percent = 0.1
            margin_h = (info["xmax"] - info["xmin"]) * margin_percent
            margin_w = (info["ymax"] - info["ymin"]) * margin_percent

        crop_xmin = max(0, int(info["xmin"] - margin_w))
        crop_xmax = min(info["image_width"], int(info["xmax"] + margin_w))
        crop_ymin = max(0, int(info["ymin"] - margin_h))
        crop_ymax = min(info["image_height"], int(info["ymax"] + margin_h))

        info["crop_xmin"], info["crop_xmax"] = crop_xmin, crop_xmax
        info["crop_ymin"], info["crop_ymax"] = crop_ymin, crop_ymax
        info["crop_h"] = crop_ymax - crop_ymin
        info["crop_w"] = crop_xmax - crop_xmin


def min_pixel(info):
    if info["No_sku"]:
        pass
    else:
        area = info["area"]
        area.sort()
        areas_smallest_10percent = area[:int(len(area) * .1)]
        area_min = np.mean(areas_smallest_10percent)
        k_min_pixel = min(np.sqrt(float(args.min_pixel) / area_min), 1)
        info["k_min_pixel"] = k_min_pixel


def min_resize_scale(target_w, target_h, img_w, img_h):
    target_ratio = target_h / target_w
    orig_ratio = img_h / img_w
    scale = 1
    if orig_ratio > target_ratio:
        scale = target_h / img_h
    elif orig_ratio < target_ratio:
        scale = target_w / img_w
    elif orig_ratio == target_ratio:
        scale = target_w / img_w
    return scale


def check_bndboxes(json_data):
    filename = json_data["filename"]
    width = json_data["image_width"]
    height = json_data["image_height"]

    for obj in json_data.get("bndboxes"):
        box = obj
        box["w"] = float(box["w"])
        box["h"] = float(box["h"])
        box["x"] = float(box["x"])
        box["y"] = float(box["y"])

        xma = float((obj.get("x")+obj.get("w")) / width)
        yma = float((obj.get("y")+obj.get("h")) / height)
        if xma > 1.01 or yma > 1.01:
            error_str = "x: {} , y: {}, w: {}, h: {}".format(box["x"],  box["y"], box["w"], box["h"])
            print(error_str)
            error_str_2 = "width: {}, height: {}".format(width, height)
            print(error_str_2)
            raise ValueError(filename)


def remove_blackout(image, json_data, info):
    # Blackout ignore flag skus and remove nil id.
    bndboxes = json_data["bndboxes"]
    new_bndboxes = []
    black_bndboxes = []
    img_w, img_h = info["image_width"], info["image_height"]
    for bndbox in bndboxes:
        try:
            ignore = bndbox['ignore']
        except KeyError as keyErr:
            print("bnbbox {} key error in conflict annotations".format(keyErr))
            ignore = False

        if ignore:
            black_bndboxes.append(bndbox)
        elif bndbox['id'] == 'nil':
            print("ignore id nil bnbbox")
            pass
        else:
            new_bndboxes.append(bndbox)

    json_data["bndboxes"] = new_bndboxes
    for bndbox in black_bndboxes:
        x, y, w, h = int(bndbox["x"]), int(bndbox["y"]), int(bndbox["w"]), int(bndbox["h"])
        xmax, ymax = x + w, y + h
        x, y, xmax, ymax = max(0, x), max(0, y), min(xmax, img_w), min(ymax, img_h)
        blackbox = np.zeros((h, w, 3))
        image[y:ymax, x:xmax, :] = blackbox

    if DEBUG and black_bndboxes:
        print("The image height, width, channels is : {}".format(image.shape))
        cv2.namedWindow("remove_blackout", cv2.WINDOW_NORMAL)
        cv2.imshow("remove_blackout", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return image, json_data, info


def crop_json_img(image, json_data, info):
    crop_xmin, crop_xmax = info["crop_xmin"], info["crop_xmax"]
    crop_ymin, crop_ymax = info["crop_ymin"], info["crop_ymax"]
    crop_h, crop_w = info["crop_h"], info["crop_w"]

    json_data["image_width"] = info["image_width"] = crop_w
    json_data["image_height"] = info["image_height"] = crop_h
    for i in range(len(json_data["bndboxes"])):
        json_data["bndboxes"][i]["x"] = json_data["bndboxes"][i]["x"] - crop_xmin
        json_data["bndboxes"][i]["y"] = json_data["bndboxes"][i]["y"] - crop_ymin
    image = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax, :]
    if DEBUG:
        print("The image height, width, channels is : {}".format(image.shape))
        print("crop_xmin: {}, crop_xmax: {}, crop_ymin: {},"
              " crop_ymax: {}, crop_h: {}, crop_w {}"
              .format(crop_xmin, crop_xmax, crop_ymin, crop_ymax, crop_h, crop_w))
        print("All the info: {}".format(info))
        cv2.namedWindow("crop_json_img", cv2.WINDOW_NORMAL)
        cv2.imshow("crop_json_img", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return image, json_data, info


def pad_json_img(image, json_data, info):
    target_w, target_h = args.target_w, args.target_h
    img_w, img_h = info["image_width"], info["image_height"]

    delta_w = target_w - img_w
    delta_h = target_h - img_h
    top, bottom = delta_h // 2, delta_h - delta_h // 2
    left, right = delta_w // 2, delta_w - delta_w // 2
    for i in range(len(json_data["bndboxes"])):
        json_data["bndboxes"][i]["x"] = json_data["bndboxes"][i]["x"] + left
        json_data["bndboxes"][i]["y"] = json_data["bndboxes"][i]["y"] + top

    json_data["image_width"] = info["image_width"] = target_w
    json_data["image_height"] = info["image_height"] = target_h
    color = [0, 0, 0]
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    if DEBUG:
        print("The image height, width, channels is : {}".format(image.shape))
        print("All the info: {}".format(info))
        cv2.namedWindow("pad_json_img", cv2.WINDOW_NORMAL)
        cv2.imshow("pad_json_img", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return image, json_data, info


def resize_json_img(image, json_data, info):
    target_w, target_h = args.target_w, args.target_h
    img_w, img_h = info["image_width"], info["image_height"]
    crop_w, crop_h = info["crop_w"], info["crop_h"]

    k_resize = min_resize_scale(target_w, target_h, crop_w, crop_h)
    k_resize = min(1.0, k_resize)
    info["k_resize"] = k_resize
    json_data["image_width"] = info["image_width"] = new_w = int(img_w * k_resize)
    json_data["image_height"] = info["image_height"] = new_h = int(img_h * k_resize)
    for i in range(len(json_data["bndboxes"])):
        json_data["bndboxes"][i]["x"] = int(json_data["bndboxes"][i]["x"] * k_resize)
        json_data["bndboxes"][i]["y"] = int(json_data["bndboxes"][i]["y"] * k_resize)
        json_data["bndboxes"][i]["w"] = int(json_data["bndboxes"][i]["w"] * k_resize)
        json_data["bndboxes"][i]["h"] = int(json_data["bndboxes"][i]["h"] * k_resize)
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Update Info
    get_info(json_data, info)
    get_crop_margin(info, fix_h=int(100 * k_resize), fix_w=int(100 * k_resize))
    if DEBUG:
        print("The image height, width, channels is : {}".format(image.shape))
        print("k_resize: {} , crop_h: {}, crop_w {}".format(k_resize, crop_h, crop_w))
        print("All the info: {}".format(info))
        cv2.namedWindow("resize_json_img", cv2.WINDOW_NORMAL)
        cv2.imshow("resize_json_img", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return image, json_data, info


def expand_json_img(image, json_data, info):
    target_w, target_h = args.target_w, args.target_h
    img_w, img_h = info["image_width"], info["image_height"]
    crop_xmin, crop_xmax = info["crop_xmin"], info["crop_xmax"]
    crop_ymin, crop_ymax = info["crop_ymin"], info["crop_ymax"]
    crop_h, crop_w = info["crop_h"], info["crop_w"]

    delta_h = min(target_h, img_h) - crop_h
    delta_w = min(target_w, img_w) - crop_w

    top, bottom = delta_h // 2, delta_h - delta_h // 2
    left, right = delta_w // 2, delta_w - delta_w // 2

    max_top, max_bottom = crop_ymin, img_h - crop_ymax
    max_left, max_right = crop_xmin, img_w - crop_xmax

    if top > max_top:
        delta = top - max_top
        top = max_top
        bottom = bottom + delta
    elif bottom > max_bottom:
        delta = bottom - max_bottom
        bottom = max_bottom
        top = top + delta

    if left > max_left:
        delta = left - max_left
        left = max_left
        right = right + delta
    elif right > max_right:
        delta = right - max_right
        right = max_right
        left = left + delta

    info["crop_xmin"] = crop_xmin = crop_xmin - left
    info["crop_xmax"] = crop_xmax = crop_xmax + right
    info["crop_ymin"] = crop_ymin = crop_ymin - top
    info["crop_ymax"] = crop_ymax = crop_ymax + bottom

    info["crop_h"] = crop_ymax - crop_ymin
    info["crop_w"] = crop_xmax - crop_xmin

    if DEBUG:
        cv2.namedWindow("expand_json_img", cv2.WINDOW_NORMAL)
        cv2.imshow("expand_json_img", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return image, json_data, info


def overwrite_image_json(photo_file, json_path, info):
    with open(json_path, "r") as json_file:
        json_data = json.load(json_file)
    image = cv2.imread(photo_file)

    def check():
        img_w, img_h = info["image_width"], info["image_height"]
        image_h, image_w, _ = image.shape
        if img_w != image_w or img_h != image_h:
            print(photo_file)
            print("The image height, width, channels is : {}".format(image.shape))
            print("The json height, width: {}, {}".format(img_h, img_w))
            print("either image height/width and json height/width don't match !")
    check()

    if info["No_sku"]:  # no sku
        # Resize and pad. Make the background image small/minimum size
        # Same method as to_resize_crop
        # Resize to target then pad
        target_w = args.target_w
        target_h = args.target_h
        img_w = info["image_width"]
        img_h = info["image_height"]

        # Resize
        k_resize = min_resize_scale(target_w, target_h, img_w, img_h)
        k_resize = min(1.0, k_resize)
        json_data["image_width"] = info["image_width"] = new_w = int(img_w*k_resize)
        json_data["image_height"] = info["image_height"] = new_h = int(img_h*k_resize)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Pad
        delta_w = target_w - new_w
        delta_h = target_h - new_h
        top, bottom = delta_h//2, delta_h - delta_h//2
        left, right = delta_w//2, delta_w - delta_w//2
        json_data["image_width"] = info["image_width"] = target_w
        json_data["image_height"] = info["image_height"] = target_h
        color = [0, 0, 0]
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    else:
        if args.rm_ignore:
            image, json_data, info = remove_blackout(image, json_data, info)
        if args.to_crop_sku:
            image, json_data, info = crop_json_img(image, json_data, info)
        if args.to_min_pixel:
            k_min_pixel = info["k_min_pixel"]

            json_data["image_width"] = info["image_width"] = info["image_width"] * k_min_pixel
            json_data["image_height"] = info["image_width"] = info["image_height"] * k_min_pixel
            for i in range(len(json_data["bndboxes"])):
                json_data["bndboxes"][i]["x"] = json_data["bndboxes"][i]["x"] * k_min_pixel
                json_data["bndboxes"][i]["y"] = json_data["bndboxes"][i]["y"] * k_min_pixel
                json_data["bndboxes"][i]["w"] = json_data["bndboxes"][i]["w"] * k_min_pixel
                json_data["bndboxes"][i]["h"] = json_data["bndboxes"][i]["h"] * k_min_pixel
            image = cv2.resize(image, (0, 0), fx=k_min_pixel, fy=k_min_pixel, interpolation=cv2.INTER_AREA)

        if args.to_resize_pad:
            target_w = args.target_w
            target_h = args.target_h
            img_w = info["image_width"]
            img_h = info["image_height"]

            # Resize
            k_resize = min_resize_scale(target_w, target_h, img_w, img_h)
            k_resize = min(1.0, k_resize)
            json_data["image_width"] = info["image_width"] = new_w = int(img_w * k_resize)
            json_data["image_height"] = info["image_height"] = new_h = int(img_h * k_resize)
            for i in range(len(json_data["bndboxes"])):
                json_data["bndboxes"][i]["x"] = json_data["bndboxes"][i]["x"] * k_resize
                json_data["bndboxes"][i]["y"] = json_data["bndboxes"][i]["y"] * k_resize
                json_data["bndboxes"][i]["w"] = json_data["bndboxes"][i]["w"] * k_resize
                json_data["bndboxes"][i]["h"] = json_data["bndboxes"][i]["h"] * k_resize
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Pad
            delta_w = target_w - new_w
            delta_h = target_h - new_h
            top, bottom = delta_h // 2, delta_h - delta_h // 2
            left, right = delta_w // 2, delta_w - delta_w // 2
            json_data["image_width"] = info["image_width"] = target_w
            json_data["image_height"] = info["image_height"] = target_h
            for i in range(len(json_data["bndboxes"])):
                json_data["bndboxes"][i]["x"] = json_data["bndboxes"][i]["x"] + left
                json_data["bndboxes"][i]["y"] = json_data["bndboxes"][i]["y"] + top
            color = [0, 0, 0]
            image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        if args.to_crop_resize_pad:
            target_w, target_h = args.target_w, args.target_h
            img_w, img_h = info["image_width"], info["image_height"]
            crop_w, crop_h = info["crop_w"], info["crop_h"]

            if target_w > img_w or target_h > img_h:
                # print(photo_file)
                # C:\Users\wen.xiang.chew\Desktop\dataset\train_resized\photos/NL500148_8.jpg
                # only one that fails with width shorter.
                if target_w > img_w and target_h > img_h:
                    # just pad
                    image, json_data, info = pad_json_img(image, json_data, info)
                else:  # target_w > img_w or target_h > img_h
                    if crop_h > target_h or crop_w > target_w:
                        # resize according to height (crop_h == target_h)
                        image, json_data, info = resize_json_img(image, json_data, info)
                    else:  # crop_h < target_h or # crop_w < target_w
                        # expand (crop_h == target_h or crop_w == target_w )
                        image, json_data, info = expand_json_img(image, json_data, info)
                    image, json_data, info = crop_json_img(image, json_data, info)
                    # image, json_data, info = pad_json_img(image, json_data, info)
            else:
                if target_w > crop_w and target_h > crop_h:
                    # expand max(crop, target)
                    # expand/2 around crop x y
                    image, json_data, info = expand_json_img(image, json_data, info)
                    image, json_data, info = crop_json_img(image, json_data, info)
                else:  # target_w < crop_w or target_h < crop_h:
                    # Use min_resize_scale? (fit either h or w)
                    # expand the other dimension. (if can?)
                    image, json_data, info = resize_json_img(image, json_data, info)
                    image, json_data, info = expand_json_img(image, json_data, info)
                    image, json_data, info = crop_json_img(image, json_data, info)
                    # image, json_data, info = pad_json_img(image, json_data, info)
        check_bndboxes(json_data)

    height, width, channels = image.shape
    img_w = info["image_width"]
    img_h = info["image_height"]
    # Could use Assert
    if img_h == height == args.target_h or img_w == width == args.target_w:
        pass
    else:
        print(photo_file)
        print("Final Height: %i and Final Width: %i" % (height, width))

    if DEBUG:
        cv2.namedWindow("Final", cv2.WINDOW_NORMAL)
        cv2.imshow("Final", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    #  To overwrite image and json
    WRITE = False
    if WRITE:
        with open(json_path, "w") as json_file:
            json.dump(json_data, json_file, indent=4)
        cv2.imwrite(photo_file, image)


if __name__ == '__main__':
    start_time = time.time()

    input_folder = abspath(args.input_folder)
    if args.out_folder is None:
        out_folder = abspath(args.input_folder+"_resized")
    else:
        out_folder = abspath(args.out_folder)
    print("new dataset saved to ", out_folder)

    try:
        print("Attempting to copy folder.")
        start_time = time.time()
        copytree(input_folder, out_folder)
        end_time = time.time()
        print("Elapsed Time to copy: {:.2f}".format(end_time - start_time))
    except Exception as e:
        # print(e)
        print("Folder already exist")
        print("Ignore and convert images and annotations.")

    # find the photo files and annotation files
    files = find_files(out_folder)
    total = len(files)
    counter = 0

    # TRY THREAD - START
    def worker(photo_path, json_path, info):
        overwrite_image_json(photo_path, json_path, info)
    running_threads = []
    lock = threading.Lock()
    # TRY THREAD - END

    print("Start iterating over files. Total files: {}".format(total))
    start_time = batch_time = time.time()
    for (photo_path, json_path) in files:
        # json_path = r'C:\\Users\\wen.xiang.chew\\Desktop\\dataset\\train_resized\\photos\\Annotations\\74700058_iPhone6_4.jpg.json'
        # photo_path = r'C:\\Users\\wen.xiang.chew\\Desktop\\dataset\\train_resized\\photos\\74700058_iPhone6_4.jpg'
        with open(json_path, "r") as json_file:
            json_data = json.load(json_file)
        info = {}
        get_info(json_data, info)

        if args.to_crop_sku:
            get_crop_margin(info)
        if args.to_min_pixel:
            min_pixel(info)
        if args.to_resize_pad:
            get_crop_margin(info)
        if args.to_crop_resize_pad:
            get_crop_margin(info)

        # TRY THREAD - START
        short_name = basename(photo_path)
        t = threading.Thread(name="thread for " + short_name, target=worker,
                             args=(photo_path, json_path, info))
        running_threads.append(t)
        t.start()

        if len(running_threads) >= 8:
            for t_idx, t in enumerate(running_threads):
                t.join()
            # print("8 images has been resized!")
            running_threads = []
        # TRY THREAD - END

        # overwrite_image_json(photo_path, json_path, info)

        counter += 1
        progress = counter/total
        done = int(progress*100)
        if counter % 20 == 0:
            end_time = time.time()
            batch_time = end_time - batch_time
            print("{}% Completed: {}/{} , Batch Time: {:.2f} , Elapsed Time: {:.2f}".format(
                    done, counter, total, batch_time, end_time - start_time))
            batch_time = end_time

    # TRY THREAD - STARTs
    for t in running_threads:
        t.join()
    running_threads = []
    # TRY THREAD - END

    end_time = time.time()
    print("Elapsed Time: {:.2f}".format(end_time - start_time))
    print("End")
