import torch.utils.data
from dataloaders.base_data_loader import BaseDataLoader
import copy


def CreateDataLoader(opt):
    data_loader = MyDatasetDataLoader()
    data_loader.initialize(opt)
    return data_loader


def CreateValDataLoader(opt):
    val_opt = copy.deepcopy(opt)
    val_data_loader = MyDatasetDataLoader()
    val_opt.phase = "val"
    val_data_loader.initialize(val_opt, is_Training=val_opt.isTrain)
    return val_data_loader


def CreateDataset(opt):
    dataset = None
    if opt.dataset_mode == 'lpba40_contrastive_learning':
        from dataloaders.lpba40_dataloader_contrastive_learning import LPBA40
        dataset = LPBA40()
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("datasets [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class MyDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt, is_Training=True):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        if is_Training:
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batchSize,
                shuffle=True,
                num_workers=int(opt.nThreads))
        else:
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=1,
                shuffle=False,
                num_workers=int(opt.nThreads))

    def load_data(self):
        return self

    def get_classes(self):
        return self.dataset.A_classes, self.dataset.A_class_to_idx

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batchSize >= self.opt.max_dataset_size:
                break
            yield data
