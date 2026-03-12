import fiftyone.zoo as foz

train = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    label_types=["detections"],
    classes=["bicycle", "car", "motorcycle", "bus", "train", "truck", "boat", "airplane", "person"],
)

val = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    label_types=["detections"],
    classes=["bicycle", "car", "motorcycle", "bus", "train", "truck", "boat", "airplane", "person"],
)

test = foz.load_zoo_dataset(
    "coco-2017",
    split="test",
    label_types=["detections"],
    classes=["bicycle", "car", "motorcycle", "bus", "train", "truck", "boat", "airplane", "person"],
)
