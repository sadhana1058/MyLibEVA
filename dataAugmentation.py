from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

class Cifar10Dataset(torchvision.datasets.CIFAR10):
    def __init__(self, root="../data", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]
        
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return image, label


def getTrainTransforms():
    train_transforms = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.CoarseDropout(max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=16, min_width=16,
                        fill_value=0.4734),
        #A.RandomCrop(32, 4),
        A.Cutout(num_holes=1, max_h_size=16,max_w_size = 16,p=1),
        A.Normalize(mean=(0.49139968, 0.48215827 ,0.44653124), std=(0.24703233,0.24348505,0.26158768)),
        ToTensorV2(),
         ],
    p=1.0
)
    return train_transforms

def getTestTransforms():
    test_transforms = A.Compose([
         A.Normalize(mean=(0.49139968, 0.48215827 ,0.44653124), std=(0.24703233,0.24348505,0.26158768)),
         ToTensorV2(),
         ])

    return test_transforms
