import fiftyone.zoo as foz

train = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    label_types=["detections"],
    classes=["person"],
)

val = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    label_types=["detections"],
    classes=["person"],
)

test = foz.load_zoo_dataset(
    "coco-2017",
    split="test",
    label_types=["detections"],
    classes=["person"],
)
