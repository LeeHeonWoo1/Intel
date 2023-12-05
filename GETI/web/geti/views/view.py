from flask import Blueprint, render_template, url_for, request, flash
from werkzeug.utils import redirect

from geti import db
from geti.models import User
import hashlib

# set FLASK_APP=geti
# set FLASK_DEBUG=true

bp = Blueprint('main', __name__, url_prefix='/')

@bp.route('/')
def hello_pybo():
    return 'web of GETI Project'

@bp.route("/predict")
def prediction():
    return "will be prediction page"

@bp.route("/sign_up")
def sign_up():
    return render_template("user/signUp.html")

# 암호 해쉬 이후 비교가 안됨. 어떻게 해결할지 찾을 것
@bp.route("/sign_upload", methods=['POST',])
def sign_upload():
    username = request.form.get('username')
    userid = request.form.get('userid')
    userpw = request.form.get('password')
    userphone = request.form.get('user_phone')
    
    m = hashlib.sha256()
    m.update(userpw.encode("utf-8"))
    
    user = User(username = username, userid = userid, userpw = m.hexdigest(), user_phone = userphone)
    db.session.add(user)
    db.session.commit()
    
    flash("회원가입이 완료되었습니다. 로그인 창으로 이동합니다.")
    return redirect(url_for("main.move_to_login"))

    
@bp.route("/login", methods=['POST'])
def login():
    username = request.form.get("username")
    userpw = request.form.get("password")
    
    m = hashlib.sha256()
    m.update(userpw.encode("utf-8"))
    
    db_userpw = db.session.query(User.userpw).filter(User.userid == username)
    
    if m.hexdigest() != db_userpw:
        print(m.hexdigest())
        flash("ID, 패스워드를 확인하세요.")
        return redirect(url_for("main.move_to_login"))
    else:
        flash("로그인이 완료되었습니다.")
        return render_template("user/signUp.html", {"userid":username})
    
@bp.route("/move_to_login")
def move_to_login():
    return render_template("user/login.html")