import numpy as np

dataset_path = '/home/mohbat/RoadSegmentation/DataSet'
path_ckpt = 'ckp/'
path_save = '/home/mohbat/RoadSegmentation/SegNet11/ckp'
path_train = '/home/mohbat/RoadSegmentation/DataSet/SegNet/train.txt'
path_test = '/home/mohbat/RoadSegmentation/DataSet/SegNet/test.txt'
path_val = '/home/mohbat/RoadSegmentation/DataSet/SegNet/val.txt'
path_output = 'output/'

train_iteration = 20000  # total training Iterations
save_model_itr = 500    # save checkpoint after this number of Iterations
val_iter       = 500    # validate after this number of Iterations
TRAIN_BATCH_SIZE = 10
EVAL_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1


INITIAL_LEARNING_RATE = 0.001      # Initial learning rate.


# for CamVid
IMAGE_HEIGHT = 360
IMAGE_WIDTH = 480
IMAGE_DEPTH = 3
NUM_CLASSES = 11

MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.

classes_colors =  [[128,128,128], [128,0,0], [192,192,128], [255,69,0], [128,64,128], [60,40,222], [128,128,0],
[192,128,128], [64,64,128], [64,0,128],[64,64,0],[0,128,192], [0,0,0]]

# ===============================
Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road_marking = [255,69,0]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

label_colours = np.array([Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])
