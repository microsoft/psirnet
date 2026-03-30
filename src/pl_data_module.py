import lightning as L
from torch.utils.data import DataLoader
from .data import SliceDataset, PSIRNetDataTransform, parallel_collate_fn


class PSIRNetDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_csv: str = "./csv_files/train.csv",
        val_csv: str = "./csv_files/val.csv",
        test_csv: str = "./csv_files/test.csv",
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        test_batch_size: int = 8,
    ):
        super().__init__()
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.transform = PSIRNetDataTransform()

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = SliceDataset(
                self.train_csv, transform=self.transform
            )
            self.val_dataset = SliceDataset(
                self.val_csv, transform=self.transform
            )

        if stage == "test" or stage is None:
            self.test_dataset = SliceDataset(
                self.test_csv, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=0,  # Keep 0 for parallel_collate_fn
            pin_memory=True,
            collate_fn=parallel_collate_fn,
            drop_last=True,  # Drop last incomplete batch
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=parallel_collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=parallel_collate_fn,
        )
