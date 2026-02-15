from dataset.DICOMDataset import DICOMDataset
from models.USFM.USFMClip import USFMClip
import argparse
import torch
from torch.utils.data import DataLoader
import os
import random
from tqdm import tqdm
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch.optim as Optim
import torch.nn as nn

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom_dir", type=str, required=True)
    parser.add_argument("--json_dir", type=str, required=True)
    parser.add_argument("--context_window", type=int, default=5)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    return parser.parse_args()

def get_model(args):
    if args.model == "usfm":
        model_config_path = "models/USFM/configs/model.yaml"
        model = USFMClip(model_config_path=model_config_path, clip_size=args.context_window * 2 + 1, finetune_encoder=False)
    else:
        raise ValueError(f"Model {args.model} not supported currently.")
    
    return model

def get_transforms(args):
    if args.model == "usfm":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(size=(224,224)),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        ])
    else:
        raise ValueError(f"Model {args.model} not supported currently.")

def get_train_test_jsons(args):
    json_dir = args.json_dir

    all_json_fnames = []

    for fname in os.listdir(json_dir):
        if not fname.endswith("json"):
            continue
        all_json_fnames.append(fname)
    
    random.shuffle(all_json_fnames)

    num_train = int(len(all_json_fnames) * 0.70)
    return all_json_fnames[:num_train], all_json_fnames[num_train]

def main():
    args = get_args()
    train_jsons, test_jsons = get_train_test_jsons(args)
    
    img_transforms = get_transforms(args)

    train_ds = DICOMDataset(dicom_dir_path=args.dicom_dir, json_dir_path=args.json_dir, context_window=args.context_window, pad_mode="zero", jsons_to_include=train_jsons, transform=img_transforms)
    test_ds = DICOMDataset(dicom_dir_path=args.dicom_dir, json_dir_path=args.json_dir, context_window=args.context_window, pad_mode="zero", jsons_to_include=test_jsons, transform=img_transforms)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=1)


    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = get_model(args)
    model = model.to(device=device)

    optim = Optim.Adam(params=model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(args.epochs), desc="Model training"):
        for _, batch_data in enumerate(train_loader):
            clip = batch_data["clip"].to(device=device)
            label = batch_data["keyframe_mask"].to(device=device)

            optim.zero_grad()

            pred = model(clip)

            print(pred.shape, label.shape)
            loss = criterion(label, pred)

            loss.backward()

            optim.step()

            print("Loss:", loss)

            break

if __name__ == "__main__":
    main()