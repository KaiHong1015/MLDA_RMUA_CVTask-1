# MLDA_RMUA_CVTask
<p>Repository contains Python implemented object detection with depth recognition training and inference notebooks, weight, result videos, and learning log.</p>

## Installation
```bash
pip install -U -r requirements.txt
```
## Usage
#### Pre-saved Result Videos
* I had prepared the inference videos inside the `result_videos` folder by doing prediction on every single frame and stacking together to form a video. `Inference_Video_Generator.ipynb` notebook shows the work I done for it. Frames were extracted from all RGB videos and saved inside `video_frames_images` folder.

#### Inference on Images
* I had prepared an example on how to do inference on image data inside `predict_images.py`. Below are the variables that needed to be specified.
```python
images_root = 'data/Ally/right' ## Your Images Root Directory
depthmaps_root = 'data/Ally/depthmap' ## Your DepthMap Root Directory, and must be in the same order as the images
isleft = False ## Is it doing on left camera data?
model = get_model(weight_path='weight/fold1_resnet50_fpn_BestModel.pth') ## Your weight path
```
#### Real Time direct inference on video
* Inside `realtime_inference.py` shows an example on how to do inference directly on video. But this required better GPU to run on it or else you will end up with a low FPS performance. Below are the variables that needed to be specified. 
```python
cap = cv2.VideoCapture('data/Ally/left/left_cam.avi') ## Your Video Path
dms = np.load('video_frames_images/ally_depth_mm.npy') ## Your Related Video DepthMap Path in .npy format
isleft = True ## Is it doing on left camera data?
model = get_model(weight_path='weight/fold1_resnet50_fpn_BestModel.pth') ## Your weight path
```

## Label Image Dataset
Since the image data given is not labeled yet, thus before training, image labeling is needed. All of the images are labeled and divided into 4 classes `Ally Robot, Ally Armor, Enemy Robot, Enemy Armor`. All the labeled data are first save as `.xml` in `Pascal VOC` format, and later transformed into a dataframe `bbox_df.csv`. The transformation work is done in the `xml_to_csv.ipynb` notebook.

## Training
In this repository, Faster R-CNN model with a pre-trained ResNet-50-FPN backbone is implemented in Pytorch. Faster R-CNN was evolved from R-CNN and Fast R-CNN. The main difference between Fast R-CNN and Faster R-CNN is that the traditional Selective Search Algorithm is replaced by Regional Proposal Network which allowed faster computation on propose region. 

Before starting training our model, the library `albumentations` is used to augment our images dataset as it provide greater and powerful augmentation choices. Below shows some examples of augmented images.

<div align="center"><img src=https://user-images.githubusercontent.com/84235717/120287200-a10fb480-c2f1-11eb-9efb-db89d04559a8.png height="15%" width="70%"></div>

For better training and validation purpose, stratifiedkfold with 6 folds is used to split our data with same percentage on each class. As there are 6 folds, which means there will result in 6 difference weights on each fold, but we only provide the best weight (fold 1) in this repository. You can also find the learning curve and learning result for each epoch of fold 1 weight in `learning_logs` folder.

In training loop, `Averager()` is used to calculate the average loss of our model as the model will return a dictionary with 4 difference losses. The averaged validation loss was tracked and the epoch with the lowest averaged validation loss, its weight will be saved. As pre-trained model was used, the losses started to converged at around epoch 10, and each training fold taked 5-10 minutes with CUDA. Below are the parameters that I used for training.
```bash
lr = 1e-5
epochs = 20
optimizer = Adam
weight_decay = 1e-7
clip_grad_value_ = 0.1
lr_scheduler = ReduceLROnPlateau
```
## Inference
For better visualisation purpose I had generated 4 videos based on left, right cam of ally and enemy, the videos can be found in `result_videos` folder. The videos are generated by making prediction on every frame and stacking frames to video. I also annotated the predicted bouding boxes with corresponding distance through depthmap data provided. As the depthmap is not aligned with the RGB data, so I had added a variable `c` to calibrate the left, right cameras. `predict_images.py` and `realtime_inferecne.py` are also provided in this repository to predict images or do real time object detection on video or camera.

Initially, my model made some silly predictions. Since there exists something similar to the robot's armor at the upper part of the robot, but it is not the armor. Therefore, I implemented `get_valid_prediction()` to select those boxes having `xmin` at the bottom part of the robot. Futhermore, as model will predict several similar bounding boxes on a single target, `run_wbf()` is implemented to fuse these boxes into one. Below shows the improvement that I had made.

<h2 align="center">
  Example
</h2>

Before            |  After
:-------------------------:|:-------------------------:
![](https://user-images.githubusercontent.com/84235717/120179045-b37be680-c23c-11eb-8909-ca317ed4c6a6.gif)  |  ![](https://user-images.githubusercontent.com/84235717/120179231-e0c89480-c23c-11eb-90be-b42a351b1e5c.gif)

## Citation
### Weighteb Boxes Fusion
```bash
@article{solovyev2021weighted,
  title={Weighted boxes fusion: Ensembling boxes from different object detection models},
  author={Solovyev, Roman and Wang, Weimin and Gabruseva, Tatiana},
  journal={Image and Vision Computing},
  pages={1-6},
  year={2021},
  publisher={Elsevier}
}
```
### Albumentations
```bash
@Article{info11020125,
    AUTHOR = {Buslaev, Alexander and Iglovikov, Vladimir I. and Khvedchenya, Eugene and Parinov, Alex and Druzhinin, Mikhail and Kalinin, Alexandr A.},
    TITLE = {Albumentations: Fast and Flexible Image Augmentations},
    JOURNAL = {Information},
    VOLUME = {11},
    YEAR = {2020},
    NUMBER = {2},
    ARTICLE-NUMBER = {125},
    URL = {https://www.mdpi.com/2078-2489/11/2/125},
    ISSN = {2078-2489},
    DOI = {10.3390/info11020125}
}
```

