import fiftyone as fo
import fiftyone.zoo as foz

CLASSES = ["car", "motorcycle", "bus", "truck", "train", "airplane", "boat", "person"]

train = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    classes=CLASSES
)

val = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    classes=CLASSES
)

train.export(
    export_dir="/home/ss/Kirti/fiftyone/coco-2017",
    dataset_type=fo.types.YOLOv5Dataset,
    label_field="ground_truth",
    split="train",
    classes=CLASSES
)

val.export(
    export_dir="/home/ss/Kirti/fiftyone/coco-2017",
    dataset_type=fo.types.YOLOv5Dataset,
    label_field="ground_truth",
    split="val",
    classes=CLASSES
)

print("Export complete.")
