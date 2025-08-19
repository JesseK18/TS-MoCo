from datasets.ucr_dataset import UCRDataModule

def data_loader(config):
    """
    Returns a LightningDataModule with fixed splits for UCR.
    Matches the user's existing pipeline structure.
    """
    dm = UCRDataModule(
        data_dir=config['data_dir'],
        dataset_name=config['dataset'],
        batch_size=config.get('batch_size', 16),
        num_workers=config.get('num_workers', 0),
        q_split=config.get('q_split', 0.15),
        seed=config.get('seed', 18),
        permute_indexes=config.get('permute_indexes', False)
    )
    dm.prepare_data()
    dm.setup()
    return dm
