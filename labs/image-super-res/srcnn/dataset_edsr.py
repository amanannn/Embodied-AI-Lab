import random
from PIL import Image
from torch.utils.data import Dataset
from utils import image_to_tensor, list_image_files


class EDSRDataset(Dataset):
    def __init__(self, hr_dir, scale=4, patch_size=192, train=True, samples_per_epoch=2000):
        self.image_files = list_image_files(hr_dir)
        self.scale = scale
        self.patch_size = patch_size
        self.train = train
        self.samples_per_epoch = samples_per_epoch

        if len(self.image_files) == 0:
            raise RuntimeError(f"No image files found in {hr_dir}")
        if patch_size % scale != 0:
            raise ValueError(f"patch_size ({patch_size}) must be divisible by scale ({scale})")

        self.images = []
        print(f"Loading {len(self.image_files)} HR images into memory...")

        for path in self.image_files:
            img = Image.open(path).convert("RGB")
            img = self._crop_to_multiple(img)
            self.images.append({"name": path.stem, "image": img})

        print("Finished loading images.")

    def __len__(self):
        if self.train:
            return self.samples_per_epoch
        return len(self.images)

    def _crop_to_multiple(self, image):
        w, h = image.size
        w = w - w % self.scale
        h = h - h % self.scale
        return image.crop((0, 0, w, h))

    def _random_crop(self, image):
        w, h = image.size
        if w < self.patch_size or h < self.patch_size:
            scale = self.patch_size / min(w, h)
            w = max(self.patch_size, int(w * scale))
            h = max(self.patch_size, int(h * scale))
            image = image.resize((w, h), Image.BICUBIC)

        left = random.randint(0, w - self.patch_size)
        top = random.randint(0, h - self.patch_size)
        return image.crop((left, top, left + self.patch_size, top + self.patch_size))

    def _make_lr(self, hr):
        w, h = hr.size
        lr_w, lr_h = w // self.scale, h // self.scale
        return hr.resize((lr_w, lr_h), Image.BICUBIC)

    def __getitem__(self, index):
        item = self.images[index % len(self.images)]
        hr = item["image"].copy()

        if self.train:
            hr = self._random_crop(hr)

        lr = self._make_lr(hr)

        return {
            "lr": image_to_tensor(lr),
            "hr": image_to_tensor(hr),
            "name": item["name"],
        }
