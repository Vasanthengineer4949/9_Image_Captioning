import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageCaptioningDataset(Dataset):
    def __init__(
        self, root_dir, df, feature_extractor, tokenizer, max_target_length=512
    ):
        self.root_dir = root_dir
        self.df = df
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.max_length = max_target_length
        self.excep_lst = []

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        try:
            image_path = self.df["images"][idx]
            text = self.df["text"][idx]
        except:
            image_path = self.df["images"][idx+1]
            text = self.df["text"][idx+1]
            self.excep_lst.append(idx)
            print(len(self.excep_lst), "Passed", idx)            
        image = Image.open(self.root_dir + "/" + image_path).convert("RGB")
        pixel_values = self.feature_extractor(image, return_tensors="pt").pixel_values
        captions = self.tokenizer(
            text, padding="max_length", max_length=self.max_length
        ).input_ids
        captions = [
            caption if caption != self.tokenizer.pad_token_id else -100
            for caption in captions
        ]
        encoding = {
            "pixel_values": pixel_values.squeeze(),
            "labels": torch.tensor(captions),
        }
        return encoding


def load_dataset(root_dir, df, feature_extractor, tokenizer, max_target_length=512):
    train_dataset = ImageCaptioningDataset(
        root_dir, df, feature_extractor, tokenizer, max_target_length
    )
    return train_dataset