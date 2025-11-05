
### 📘 `src/datasets.py`

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

class UnpairedAB(Dataset):
    """Unpaired dataset for CycleGAN training and testing."""
    def __init__(self, a_dir, b_dir, size=256):
        self.a = sorted(list(Path(a_dir).glob("*")))
        self.b = sorted(list(Path(b_dir).glob("*")))
        self.ta = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize([0.5]*3,[0.5]*3)])
        self.tb = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize([0.5]*3,[0.5]*3)])
    def __len__(self): return max(len(self.a), len(self.b))
    def __getitem__(self, i):
        a = Image.open(self.a[i % len(self.a)]).convert("RGB")
        b = Image.open(self.b[i % len(self.b)]).convert("RGB")
        return self.ta(a), self.tb(b)
