import io
import json
import os
#from torchvision import models
from flask import Flask, jsonify, request
import ultralytics
from ultralytics import YOLO
import cv2
from werkzeug.utils import secure_filename
from flask_cors import CORS
import ssl 

import binascii
from flask import Flask, request, render_template, redirect, url_for, session, send_file, render_template_string



random_number = os.urandom(24)
random_hex = binascii.hexlify(random_number)
print(random_hex)

# initializing
app = Flask(__name__)

CORS(

    app,
    resources={r'*': {'origins': ['http://localhost:3000', 'https://www.cooksave.co.kr']}},
    supports_credentials=True
)

FoodList={'beet':'비트', 'bell_pepper':'파프리카', 'cabbage':'양배추', 'carrot':'당근',
'cucumber':'오이', 'egg':'계란', 'eggplant':'가지', 'garlic':'마늘', 'onion':'양파',
'potato':'감자','tomato':'토마토','zucchini':'애호박'}

IconMap={'defalut':1,'파프리카':9,'양배추':2, '당근':4, '오이':5, '계란':6, '가지':11,'마늘':8,
'양파':3, '감자':7, '토마토':12, '애호박':10,'비트':13}

def transform_labels(translated_labels):
    result = []
    label_count = {}

    # 라벨의 등장 횟수 세기
    for label in translated_labels:
        if label in label_count:
            label_count[label] += 1
        else:
            label_count[label] = 1

    # 결과 형식으로 변환
    for label, count in label_count.items():
        result.append({"label": 'label', "count": count, "icon": IconMap.get(label, None)})

    return result

# @app.before_request
# def before_request():
#     if request.url.startswith('http://'):
#         url = request.url.replace('http://', 'https://', 1)
#         code = 301
#         return redirect(url, code=code)


@app.route('/detection', methods=['GET','POST'])
def predict():
    output_labels=[]
    result_dic={}
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)

        os.makedirs('./input_dir', exist_ok=True)
        file.save(os.path.join('./input_dir', filename))
        train_img = './input_dir/' + file.filename

        #model = YOLO('./weights/best.pt') 
        model = YOLO('./weights/best_weights.pt') 
        
        results=model.predict(train_img, save=True, imgsz=320, conf=0.5)

        # label 인식값 반환
        for box in results[0].boxes:
            output_labels.append(model.names.get(box.cls.item()))
        
        translated_labels = [FoodList[label] if label in FoodList else label for label in output_labels]
        
        # 라벨 변환
        transformed_labels = transform_labels(translated_labels)

    return jsonify(transformed_labels), 200


if __name__ == '__main__':
    #ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS)
    #ssl_context.load_cert_chain(certfile='newcert.pem', keyfile='newkey.pem', password='secret')
    #ssl_context.load_cert_chain(certfile='cert.pem', keyfile='key.pem')
    #app.debug=True
    #app.run(host="0.0.0.0", port=443, ssl_context=ssl_context)
    app.run(host='0.0.0.0' )
