import numpy as np
import json
from keras.utils.data_utils import get_file

V2_LABELS_PATH = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_labels_v2.npy'


RESNET50_WEIGHTS_PATH = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_resnet50.h5'
RESNET50_WEIGHTS_PATH_NO_TOP = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_notop_resnet50.h5'

VGGFACE_DIR = 'models/vggface'

fpath = get_file('rcmalli_vggface_labels_v2.npy',
                             V2_LABELS_PATH,
                             cache_subdir=VGGFACE_DIR)
LABELS = np.load(fpath)
result = [str(LABELS[i].encode('utf8').decode('utf-8'))  for i in range(len(LABELS))]
labels_enumerated = dict(zip(result, range(len(result))))
with open("labels.json", "w") as f:
    json.dump(labels_enumerated, f)