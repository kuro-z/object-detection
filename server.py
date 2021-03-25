import os
import cv2
import time
import yaml
import uuid
import shutil
from datetime import timedelta
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from detector import Detector

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
# 解决缓存刷新问题
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(hours=1)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
with open('config.yaml', 'r', encoding='utf-8')as f:
    cfg = yaml.load(f.read(), Loader=yaml.FullLoader)


@app.route('/')
def pandaIndex():
    return render_template('Index.html')


def allowed_file(fname):
    return '.' in fname and fname.rsplit('.', 1)[1].lower() in cfg['ALLOWED_EXTENSIONS']


@app.route('/detect', methods=['POST', 'GET'])
def detect():
    file = request.files['file']
    # print(file.filename)
    if file and allowed_file(file.filename):
        ext = file.filename.rsplit('.', 1)[1]
        # 生成随机文件名
        random_name = '{}.{}'.format(uuid.uuid4().hex, ext)
        # print(random_name)
        savepath = os.path.join(cfg['CACHE_FOLDER'], secure_filename(random_name))
        result_path = os.path.join(cfg['RESULTS_FOLDER'], secure_filename(random_name))
        file.save(savepath)
        shutil.copy(savepath, os.path.join('static', savepath))
        # time-1
        t1 = time.time()
        img = cv2.imread(savepath)
        status, img_result, img_info = detector.detect(img)
        # time-2
        t2 = time.time()

        if status == 1:
            cv2.imwrite(result_path, img_result)
            shutil.copy(result_path, os.path.join('static', result_path))
        """
        status: 1 - 成功检测到, 2 - 未检测到
        """
        return jsonify({
            'status': status,
            'img_url': os.path.join('static', savepath),
            'result_url': os.path.join('static', result_path),
            'img_info': img_info,
            'time': '{:.4f}s'.format(t2-t1)
        })

    return jsonify({'status': 0})


if __name__ == '__main__':
    detector = Detector(img_size=cfg['IMG_SIZE'], threshold=cfg['THRESHOLD'], weights=cfg['WEIGHT_PATH'])

    for folder in cfg['FOLDER']:
        os.makedirs(folder, exist_ok=True)
        os.makedirs(os.path.join('static', folder), exist_ok=True)

    app.run(host=cfg['HOST'], port=cfg['PORT'], debug=True, threaded=True, processes=1)
