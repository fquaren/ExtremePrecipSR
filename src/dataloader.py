import os
from torch.utils.data import DataLoader


def create_dataloader(dataset, batch_size, shuffle=True, num_workers=None):
    """
    Creates a PyTorch DataLoader from a Dataset.
    """
    if num_workers is None:
        num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
        # A common practice is to use fewer workers than CPU cores to avoid oversubscription
        # especially if each worker loads large files.
        num_workers = min(
            num_workers, 8
        )  # Cap at 8 or adjust based on your system/data size

    print(f"Using {num_workers} workers for DataLoader.")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,  # Optimizes data transfer to GPU
    )
