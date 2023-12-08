from flask import Blueprint, render_template, url_for, request, flash, session, g
from werkzeug.utils import redirect
from keras.models import load_model
from urllib.parse import unquote

from geti import db
from geti.models import User
from werkzeug.utils import secure_filename
from geti.forms import UserCreateForm, UserLoginForm
import bcrypt
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

bp = Blueprint('main', __name__, url_prefix='/')
models = {
    '토마토' : r"C:\Users\OWNER\Desktop\plant dataset\models\tomato.h5",
    "감자" : r"C:\Users\OWNER\Desktop\plant dataset\models\potato.h5",
    "장미" : r"C:\Users\OWNER\Desktop\plant dataset\models\rose.h5",
    "레몬그라스" : r"",
    "목화" : r"D:\Intel\GETI\cotten_model\weights\GlobalAveragePooling2D-30.h5",
    "커피" : r"C:\Users\OWNER\Desktop\plant dataset\models\coffee.h5"
}

labels = {
    '토마토' : {},
    '감자' : {},
    '장미' : {},
    '레몬그라스' : {},
    '목화' : {0 : "진딧물", 1 : "세균성 마름병", 2 : "건강한 상태", 3 : "흰가루병", 4 : "표적반점"},
    '커피' : {},
}

def read_image(path):                                   
    gfile = tf.io.read_file(path)                       
    image = tf.io.decode_image(gfile, dtype=tf.float32) 
    return image

@bp.route('/')
def hello():
    return render_template("index.html")

@bp.route("/predict", methods=('GET','POST'))
def prediction():
    if request.method == 'POST' :
        image_path = r"D:\Intel\GETI\web\geti\static\pred_imgs"
        plant_name_ = request.get_data()
        back_pos = plant_name_.rfind(b"\r")
        front_pos = plant_name_.find(b"\r\n\r\n")
        
        with open("./plant_name.txt", "wb") as f:
            f.write(plant_name_[front_pos+4:back_pos])
            
        f.close()
        
        with open("./plant_name.txt", "rb") as f:
            plant_name = f.readline().decode("utf-8").replace("\n", "")
        
        file = request.files['file']
            
        filename = secure_filename(file.filename)
        file.save(os.path.join(image_path, filename))
        
        labels = {0 : "진딧물", 1 : "세균성 마름병", 2 : "건강한 상태", 3 : "흰가루병", 4 : "표적반점"} 
        model = load_model(models[plant_name])
        img = read_image(os.path.join(image_path, filename))
        img = tf.image.resize(img, (256, 256))
        image = np.array(img)
        image = image[:, :, :]
        test_image = image[tf.newaxis, ...]
        pred = model.predict(test_image, verbose = 0)
        
        result = labels[np.argmax(pred)]
        flash("예측이 완료되었습니다.") # values = [result, filename, plant_name]
        
        values = {
            "result" : result,
            "filename" : filename,
            "plant_name" : plant_name
        }
        return redirect(url_for('main.prediction', values = values))
        
    return render_template("functions/prediction.html")

@bp.route("/sign_up/", methods=('GET', 'POST'))
def sign_up():
    form = UserCreateForm()
    if request.method == 'POST' and form.validate_on_submit():
        user = User.query.filter_by(username = form.username.data).first()
        if not user:
            userpw = bytes(form.password1.data.encode("utf-8"))
            bytes_hashed_password = bcrypt.hashpw(password=userpw, salt=bcrypt.gensalt()).decode("utf-8")
            
            user = User(
                username=form.username.data,
                userid=form.userid.data,
                userpw=bytes_hashed_password,
                user_phone=form.phone.data
                )
            
            db.session.add(user)
            db.session.commit()
            
            flash("회원가입을 완료했습니다. 메인 페이지로 이동합니다.")
            return redirect(url_for('main.hello'))
        else:
            flash("이미 존재하는 사용자입니다.")
            return render_template("user/signUp.html")

    return render_template("user/signUp.html", form=form)

@bp.before_app_request
def load_user_logged_in():
    user_id = session.get("user_id")
    if user_id is None:
        g.user = None
    else:
        g.user = User.query.get(user_id)
    
@bp.route("/login/", methods=('GET', 'POST'))
def login():
    form = UserLoginForm()
    if request.method == 'POST' and form.validate_on_submit():
        user = User.query.filter_by(userid = form.userid.data).first()
        userpw = form.password.data
        db_userpw = user.userpw

        if not user:
            flash("존재하지 않는 사용자입니다.")
            return render_template("user/login.html")
        elif not bcrypt.checkpw(bytes(userpw.encode('utf-8')), db_userpw.encode("utf-8")):
            flash("ID, 패스워드를 확인하세요.")
            return render_template("user/login.html")
        else:
            session.clear()
            session['user_id'] = user.userid
            flash("로그인이 완료되었습니다.")
            return redirect(url_for('main.hello'))
    
    return render_template("user/login.html", form=form)

@bp.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('main.hello'))

@bp.route("/mypage/<string:userid>")
def mypage(userid):
    return userid+"의 메인페이지"