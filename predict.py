from utils import *
from dataset import *

image_root = 'data\\Ally\\left'
depthmaps_path = 'data\\ally_depth_mm.npy'
isleft = True

model = get_model('weights\\fold1_resnet50_BestModel.pth')
test_ds = TestDataset(image_root=image_root, depthmaps_path=depthmaps_path, isleft=isleft)

image, dm, isleft = test_ds[1]

print(dm)