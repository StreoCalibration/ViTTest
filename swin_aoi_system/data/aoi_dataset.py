from torch.utils.data import Dataset

class AoiDataset(Dataset):
    """
    PyTorch Dataset class for loading real, synthetic, or mixed AOI data.
    """
    def __init__(self, root_dir, data_source='mixed', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            data_source (string): 'real', 'synthetic', or 'mixed'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass