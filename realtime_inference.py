from utils import *
from dataset import *
from itertools import count

import time

cap = cv2.VideoCapture('data\\Ally\\left\\left_cam.avi') ## Your Video Path
dms = np.load('video_frames_images\\ally_depth_mm.npy') ## Your Related Video DepthMap Path in .npy format
isleft = True ## Is it doing on left camera data?

model = get_model(weight_path='weight\\fold1_resnet50_fpn_BestModel.pth') ## Your weight path

for i in count():
    start = time.time()
    ret, frame = cap.read()
    dm = dms[i]

    if not ret:
        break

    # Preprocess Frame Data (Normalize and ToTensor)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tsfm = transform['test'](image=frame)

    # Make Prediction
    prediction, np_image = make_prediction(model, tsfm['image'])

    # Box Selection
    boxes, scores, labels = run_wbf(prediction, 1280)
    boxes, scores, labels = get_valid_prediction(boxes, scores, labels)
 
    # Draw Bounding Boxes, Distances, FPS on Frame
    np_image = draw_info(np_image, dm, boxes, labels, isleft, fps=1/(time.time() - start))
    
    # Visualize Frame and Press 'q' to quit the video
    cv2.imshow('Ally Left', np_image)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()