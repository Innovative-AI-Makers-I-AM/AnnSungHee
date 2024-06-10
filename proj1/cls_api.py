import PIL.Image
from fastapi import FastAPI, File, UploadFile
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='models\\efficientnet_lite0.tflite')
options = vision.ImageClassifierOptions(
    base_options=base_options, max_results=3)
classifier = vision.ImageClassifier.create_from_options(options)

app = FastAPI()

import io
import PIL
import numpy as np

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    # STEP 3: Load the input image. 추론시킬 데이터 가져오고
    # image = mp.Image.create_from_file(byte_file)
    # 해당 파일은 이미지 파일이 아니라 텍스트 파일로 넘어오게 됨
    byte_file = await file.read()
    # 이미지로 읽을 수 있는 바이너리 파일로 변경                # convert char array to binary array
    image_bin = io.BytesIO(byte_file)                       # create PIL Image from binary array
    # 바이너리 파일을 파이썬이 읽을 수 있는 이미지 파일로 변경  # Convert MP Image from PIL IMAGE
    pil_img = PIL.Image.open(image_bin)

    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(pil_img))
    # STEP 4: Classify the input image. 데이터 처리하고
    classification_result = classifier.classify(image)

    # STEP 5: Process the classification result. In this case, visualize it. 결과보기
    # top_category = classification_result.classifications[0].categories[0]
    # second_category = classification_result.classifications[0].categories[1]
    # third_category = classification_result.classifications[0].categories[2]
    
    count = 3
    result_array = []
    for i in range(count):
        temp_result = classification_result.classifications[0].categories[i]
        result_array.append({"category : ":temp_result.category_name, "score : ":temp_result.score})



    # result = (f"{top_category.category_name} ({top_category.score:.2f})")
    # print(result)
    # print(top_category)
    # print(second_category)
    # print(third_category)
    #  print(result)

    return {"result" : result_array}