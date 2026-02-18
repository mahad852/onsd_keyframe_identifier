from dataset.SegmentationDataset import DICOMSegmentationDataset
import argparse
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import random
from torch.utils.data import DataLoader
import torch
import os
import torchvision.transforms.functional as F
import numpy as np
from models.USFM.models import build_seg_model
import yaml

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom_dir", type=str, required=True)
    parser.add_argument("--json_dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)  
    parser.add_argument("--num_workers", type=int, default=1)

    return parser.parse_args()


class MaskResizeOnly:
    def __init__(self, size=(224,224)):
        self.size = size

    def __call__(self, mask_pil):
        
        mask_pil = F.resize(mask_pil, self.size, interpolation=transforms.InterpolationMode.NEAREST)
        return torch.from_numpy(np.array(mask_pil)).long()

def get_image_transforms(args):
    if args.model == "usfm":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(224,224)),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        ])
    else:
        raise ValueError(f"Model {args.model} not supported currently.")
    
def get_mask_transforms(args):
    if args.model == "usfm":
        return transforms.Compose([
            MaskResizeOnly(size=(224, 224))
        ])
    else:
        raise ValueError(f"Model {args.model} not supported currently.")
    
def get_model(args, num_classes):
    if args.model == "usfm":
        model_config_path = "models/USFM/configs/SegVit.yaml"

        with open(model_config_path, "r") as f:
            config = yaml.safe_load(f)

        config["model"]["model_cfg"]["decode_head"]["num_classes"] = num_classes
        config["model"]["model_cfg"]["decode_head"]["loss_decode"]["num_classes"] = num_classes

        config["model"]["model_cfg"]["decode_head"]["img_size"] = 224
        config["model"]["model_cfg"]["backbone"]["img_size"] = 224

        model = build_seg_model(config["model"]["model_cfg"])
    else:
        raise ValueError(f"Model {args.model} not supported currently.")

    return model

def get_train_test_jsons(args):
    all_json_fnames = [f for f in os.listdir(args.json_dir) if f.endswith(".json")]
    random.shuffle(all_json_fnames)

    num_train = int(len(all_json_fnames) * 0.70)
    train = all_json_fnames[:num_train]
    test = all_json_fnames[num_train:]
    return train, test

def main():
    args = get_args()

    metadata_file = os.path.join(args.json_dir, ".v7", "metadata.json")

    train_jsons, test_jsons = get_train_test_jsons(args)

    img_transforms = get_image_transforms(args)
    mask_transforms = get_mask_transforms(args)

    train_ds = DICOMSegmentationDataset(
        dicom_dir_path=args.dicom_dir,
        json_dir_path=args.json_dir,
        metadata_file=metadata_file,
        jsons_to_include=train_jsons,
        transform=img_transforms,
        mask_transform=mask_transforms
    )
    test_ds = DICOMSegmentationDataset(
        dicom_dir_path=args.dicom_dir,
        json_dir_path=args.json_dir,
        metadata_file=metadata_file,
        jsons_to_include=test_jsons,
        transform=img_transforms,
        mask_transform=mask_transforms
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = get_model(args, num_classes=train_ds.get_num_classes())
    model = model.to(device=device)

    print(train_ds.get_num_classes(), train_ds.get_class_name_to_id(), len(train_ds))
    print(test_ds.get_num_classes(), test_ds.get_class_name_to_id(), len(test_ds))
    
    n = 0
    for b in train_loader:
        imgs = b["image"].to(device=device)
        masks = b["mask"].to(device=device)

        # for mask in masks:
        #     print(mask.unique(), mask.sum())

        output = model(imgs)

        print(output.shape)
        
        n += 1

        if n >= 20:
            break

if __name__ == "__main__":
    main()
