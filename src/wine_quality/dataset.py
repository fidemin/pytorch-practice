from torch.utils.data import Dataset


class WineDataset(Dataset):
    def __init__(self, x, t):
        super().__init__()
        self.x = x
        self.t = t

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.t[i]
