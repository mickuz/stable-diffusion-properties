import os
import argparse
import pandas as pd

from PIL import Image
from datasets import load_dataset
from torch.utils.data import Dataset


class DogCatDataset(Dataset):
    def __init__(self, data_dir, prompt, transforms=None):
        self.data_dir = data_dir
        self.transforms = transforms
        self.metadata_df = self._create_metadata_df(prompt)

    def __getitem__(self, idx):
        filename = self.metadata_df.loc[idx]["file_name"]
        prompt = self.metadata_df.loc[idx]["text"]
        path = os.path.join(self.data_dir, filename)
        image = Image.open(path)

        if self.transforms:
            image = self.transforms(image)

        return image, prompt

    def __len__(self):
        return len(self.metadata_df)

    def _create_metadata_df(self, prompt):
        filenames = os.listdir(self.data_dir)
        labels = [filename.split(".")[0] for filename in filenames]
        prompts = [prompt + label for label in labels]

        df = pd.DataFrame({"file_name": filenames, "text": prompts})

        return df

    def save_metadata_df(self, fmt="csv"):
        path = os.path.join(self.data_dir, "metadata." + fmt)
        if os.path.exists(path):
            return

        if fmt == "csv":
            self.metadata_df.to_csv(path, index=False)
        elif fmt == "json":
            self.metadata_df.to_json(path, index=False)
        else:
            raise ValueError("Not supported file format")

    def push_to_hugging_face(self, repo_name):
        dataset = load_dataset("imagefolder", data_dir=self.data_dir)
        dataset.push_to_hub(repo_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--repo_name", required=True)
    args = parser.parse_args()

    dataset = DogCatDataset(args.data_dir, prompt=args.prompt)
    dataset.save_metadata_df()
    dataset.push_to_hugging_face(args.repo_name)
