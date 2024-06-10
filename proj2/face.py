# STEP 1
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

# STEP 2
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# STEP 3
#img = ins_get_image('t1')
img = cv2.imread('twins.jpg', cv2.IMREAD_COLOR)
# img2 = cv2.imread('k2.jpg', cv2.IMREAD_COLOR)

# STEP 4
faces1 = app.get(img)
# faces2 = app.get(img2)

# STEP 5
# then print all-to-all face similarity
feat1 = np.array(faces1[0].normed_embedding, dtype=np.float32)
feat2 = np.array(faces1[1].normed_embedding, dtype=np.float32)

sims = np.dot(feat1, feat2.T)
print(sims)

