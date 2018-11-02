from dlvc.datasets.pets import PetsDataset
from dlvc.dataset import Subset

dataset = PetsDataset('/Users/mmatak/dev/college/DLVC/cifar-10/cifar-10-batches-py', Subset.TEST)
print(len(dataset))