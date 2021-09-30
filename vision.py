from pathlib import Path
from typing import Any
from typing import Callable
from typing import Optional
from typing import Union

from PIL import Image
from torch.utils.data.dataset import Dataset


class HistologyPatchDataset(Dataset):
    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable[[Any], Any]] = None,
    ):
        self.root = Path(root)
        self.transform = transform

        self.slides = sorted(Path(self.root).glob("*"))
        self.images = sorted(Path(self.root).glob("*/*"))

    def __repr__(self) -> str:
        exts = {p.suffix for p in self.images}
        s = (
            f"Dataset {self.__class__.__name__}"
            f"\n  Root: {self.root}"
            f"\n  Number of samples: {len(self)}"
            f"\n  Number of slides: {len(self.slides)}"
            f"\n  Image extensions: {', '.join(exts)}"
        )
        return s

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image_path = self.images[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image
