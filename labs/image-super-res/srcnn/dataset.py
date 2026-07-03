import random
from PIL import Image
from torch.utils.data import Dataset
from utils import image_to_tensor, list_image_files

class SRDataset(Dataset):
    def __init__(self, hr_dir, scale=2, patch_size=96, train=True, samples_per_epoch=2000):
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
            image = Image.open(path).convert("RGB")
            image = self.crop_to_multiple(image)
            self.images.append(
                {
                    "name": path.stem,
                    "image": image,
                }
            )
        
        print("Finished loading images.")

    def __len__(self):
        if self.train:
            return self.samples_per_epoch
        
        return len(self.images)

    def crop_to_multiple(self, image):
        width, height = image.size
        width = width - width % self.scale
        height = height - height % self.scale

        return image.crop((0, 0, width, height))

    def random_crop(self, image):
        width, height = image.size
        if width < self.patch_size or height < self.patch_size:
            scale = self.patch_size / min(width, height)
            new_width = max(self.patch_size, int(width * scale))
            new_height = max(self.patch_size, int(height * scale))
            image = image.resize((new_width, new_height), Image.BICUBIC)
            width, height = image.size
        
        left = random.randint(0, width - self.patch_size)
        top = random.randint(0, height - self.patch_size)
        right = left + self.patch_size
        bottom = top + self.patch_size

        return image.crop((left, top, right, bottom))
    
    def make_lr_input(self, hr):
        width, height = hr.size
        lr_width = width // self.scale
        lr_height = height // self.scale
        lr = hr.resize((lr_width, lr_height), Image.BICUBIC)
        lr_up = lr.resize((width, height), Image.BICUBIC)

        return lr_up
    
    def __getitem__(self, index):
        item = self.images[index % len(self.images)]
        hr = item["image"].copy()

        if self.train:
            hr = self.random_crop(hr)

        lr_up = self.make_lr_input(hr)

        return {
            "lr": image_to_tensor(lr_up),
            "hr": image_to_tensor(hr),
            "name": item["name"],
        }