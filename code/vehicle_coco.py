import fiftyone as fo

# Define the vehicle classes you are interested in
vehicle_classes = ["car", "motorcycle", "bus", "truck", "train", "airplane", "boat", "person"]

# Load the COCO-2017 dataset for the training split, including all available samples of the specified vehicle classes
dataset = fo.zoo.load_zoo_dataset(
    "coco-2017",  # COCO dataset version
    split="train",  # Can be "train", "validation", or "test"
    label_types=["detections"],  # We only need detections (bounding boxes)
    classes=vehicle_classes,  # Filter dataset to include only vehicle classes
)

# Load the COCO-2017 dataset for the validation split, including all available samples of the specified vehicle classes
dataset_val = fo.zoo.load_zoo_dataset(
    "coco-2017",
    split="validation",
    label_types=["detections"],
    classes=vehicle_classes,
)

# Load the COCO-2017 dataset for the test split, including all available samples of the specified vehicle classes
dataset_test = fo.zoo.load_zoo_dataset(
    "coco-2017",
    split="test",
    label_types=["detections"],
    classes=vehicle_classes,
)

# Check and print the summary of the datasets
print("Training Set Summary:")
print(dataset.summary())
print("Validation Set Summary:")
print(dataset_val.summary())
print("Test Set Summary:")
print(dataset_test.summary())


