
from torch.utils.data import Dataset, DataLoader

import pickle

class crypticLettersDataset(Dataset):
    """Cryptic letters dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        from createData import DATA_PATH
        self.root_dir = root_dir
        self.transform = transform
        data = pickle.load(open(DATA_PATH, 'rb'))
        self.le = data['le']
        self.data = data['data']
        self.idx = 0
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label, image = self.data[idx]
        #sample = {'image': image, 'label': label}
        image = self.transform(image)
        return label, image