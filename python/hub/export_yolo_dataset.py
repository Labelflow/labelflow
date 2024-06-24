import hub
import numpy as np
import os
import argparse


def read_yolo_boxes(fn: str, image_shape: tuple):
    """
    Function reads a label.txt YOLO file and returns a numpy array of yolo_boxes
    for the box geometry and yolo_labels for the corresponding box labels.
    args:
        fn = path to file
        image_shape = (H, W, C)
    """
    (h_image, w_image, _) = image_shape
    box_f = open(fn)
    lines = box_f.read()
    box_f.close()

    # Split each box into a separate lines
    lines_split = lines.splitlines()

    yolo_boxes = np.zeros((len(lines_split), 4))
    yolo_labels = np.zeros(len(lines_split))

    # Go through each line and parse data
    for l, line in enumerate(lines_split):
        line_split = line.split()
        x, y, w, h = (
            float(line_split[1]) * w_image,
            float(line_split[2]) * h_image,
            float(line_split[3]) * w_image,
            float(line_split[4]) * h_image,
        )
        yolo_boxes[l, :] = np.array(
            (
                x,
                h_image - (y + h),
                w,
                h,
            )
        )
        yolo_labels[l] = int(line_split[0])

    return yolo_boxes, yolo_labels


def export_yolo_dataset_hub(local_path: str, remote_path: str):
    name_dataset = os.path.basename(local_path)
    ds = hub.empty(
        remote_path, overwrite=True
    )  # Create the dataset on our dedicated s3 bucket

    # List of all images
    with open(os.path.join(local_path, "train.txt"), "r") as f:
        fn_imgs = map(
            lambda path_outside_dataset: local_path
            + path_outside_dataset.split(name_dataset)[1],
            f.read().splitlines(),
        )

    # List of all class names
    with open(os.path.join(local_path, "obj.names"), "r") as f:
        class_names = f.read().splitlines()

    with ds:
        ds.create_tensor("images", htype="image", sample_compression="jpeg")
        ds.create_tensor("labels", htype="class_label", class_names=class_names)
        ds.create_tensor("boxes", htype="bbox")

        for fn_img in fn_imgs:

            img_name = os.path.splitext(os.path.basename(fn_img))[0]
            fn_box = img_name + ".txt"
            image_sample = hub.read(fn_img)

            # Get the arrays for the bounding boxes and their classes
            yolo_boxes, yolo_labels = read_yolo_boxes(
                os.path.join(local_path, "obj_train_data", fn_box),
                image_sample.array.shape,
            )
            # Append data to tensors
            ds.images.append(image_sample)
            ds.labels.append(yolo_labels.astype(np.uint32))
            ds.boxes.append(yolo_boxes.astype(np.float32))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local-path",
        type=str,
        help="path to local yolo dataset dataset",
    )
    parser.add_argument(
        "--path",
        type=str,
        help="path to remote dataset",
    )

    args = parser.parse_args()
    export_yolo_dataset_hub(args.local_path, args.path)
