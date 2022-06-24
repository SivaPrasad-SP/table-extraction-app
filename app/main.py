from flask import Flask, request, Response
# import pytesseract
import cv2 as cv
import numpy as np
import torch
import requests
import json
# import pandas as pd

app = Flask(__name__)

# pytesseract.pytesseract.tesseract_cmd = "/app/.apt/usr/bin/tesseract"
# pytesseract.pytesseract.tesseract_cmd = "Tesseract-OCR/tesseract.exe"

model_name = 'yolo_model/best.pt' # last
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)

@app.route("/", methods=['GET', 'POST'])
def home_view():
        if request.method == 'POST':
                fs = request.files['file'] #.get('file')
                frame = cv.imdecode(np.frombuffer(fs.read(), np.uint8), cv.IMREAD_UNCHANGED)
        else:
                file_name = 'images/invo_7.jpg'
                frame = cv.imread(file_name)

        tbl_img = frame.copy()
        img = frame.copy()
        image = frame.copy()

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, frame_thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)

        h, w = frame_thresh.shape #[0], frame.shape[1]
        # print(h, w, ' : ', frame_thresh.shape)

        device = 'cpu'
        model.to(device)
        frame = [frame_thresh]
        results = model(frame_thresh)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

        arr = []

        for i in range(len(labels)):
                if labels[i] == 1:
                        row = cord[i]
                        x1, y1, x2, y2 = int(row[0]*w), int(row[1]*h), int(row[2]*w), int(row[3]*h)
                        # print(x1, y1, x2, y2)
                        arr.append([y1, y2, x1, x2])
                #cv.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        nt = len(arr)

        # img[y1:y2, x1:x2]

        areas = []
        for i in arr:
                h = i[1]-i[0]
                w = i[3]-i[2]
                ar = h*w
                areas.append(ar)
        max_area_index = np.argmax(areas)
        tbl = arr[max_area_index]

        tbl_roi = frame[0][tbl[0] : tbl[1], tbl[2] : tbl[3]]
        image = image[tbl[0] : tbl[1], tbl[2] : tbl[3]]
        tbl_img = image
        cv.imwrite('images/tbl_roi.jpg', tbl_img)

        # send image to tesseract-sp & print response.
        url = 'http://tesseract-sp.herokuapp.com/'
        files = {'file': open('images/tbl_roi.jpg', 'rb')}
        resp = requests.post(url, files=files)
        if resp.status_code == 200:
                this_response = json.loads(resp.content)
                print(type(this_response), ' << type') # dict

                tbl_rows = this_response["data"]["img2tbl"]

                # result_df = pd.DataFrame(tbl_rows[1:], columns=tbl_rows[0])
                # result_df.to_csv('outputs/result.csv', index=False)
                with open("outputs/result.csv", "w") as f:
                        for row in tbl_rows:
                                f.write("%s\n" % ','.join(str(col) for col in row))
        else:
                print(resp.content)
        # print(resp.content, '<<<<<<-----------------------')

        # 'img2txt': pytesseract.image_to_string(tbl_img),
        response_data = {'data': {'no_of_tables ': nt, 'img_shape':img.shape, 'tbl_shape': tbl_img.shape, 'table_data': tbl_rows}}
        return response_data

@app.route("/download")
def download():
    with open("outputs/result.csv") as fp:
        csv = fp.read()
    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition":
                 "attachment; filename=results.csv"})