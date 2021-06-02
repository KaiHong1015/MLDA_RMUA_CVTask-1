from utils import *
from dataset import *

images_root = 'data/Ally/right' ## Your Images Root Directory
depthmaps_root = 'data/Ally/depthmap' ## Your DepthMap Root Directory, and must be in the same order as the images
isleft = False ## Is it doing on left camera data?
model = get_model(weight_path='weight/fold1_resnet50_fpn_BestModel.pth') ## Your weight path

test_ds = TestDataset(images_root=images_root, depthmaps_root=depthmaps_root)

for ds in test_ds:
    image, dm = ds

    #Making Prediction on Image
    prediction, np_image = make_prediction(model, image)

    ## Select Valid Prediction
    boxes, scores, labels = run_wbf(prediction=prediction, image_max_size=1280)
    boxes, scores, labels = get_valid_prediction(boxes, scores, labels)

    # Draw Bounding Boxes and Distances on Image
    np_image = draw_info(np_image, dm, boxes, labels, isleft)

    # Visualize (Press 'spacebar' for next image, other key will Quit the Prediction Loop)
    cv2.imshow('Prediction', np_image)

    if cv2.waitKey(0) == ord(' '):
        continue
    else:
        break

cv2.destroyAllWindows()
