from flask import Blueprint, render_template, url_for, request, flash, session, g
from werkzeug.utils import redirect

from geti import db
from geti.models import User
from geti.forms import NewQuestionForm
from geti.models import Question, Answer
from datetime import datetime

bp_func = Blueprint('function', __name__, url_prefix='/func')

@bp_func.route("/board/")
def board():
    page = request.args.get('page', type=int, default=1)  # 페이지
    question_list = Question.query.order_by(Question.id.desc())
    question_list = question_list.paginate(page=page, per_page=20)
    return render_template("functions/board.html", question_list=question_list)

@bp_func.route("/board/search", methods=('GET', 'POST'))
def serach():
    if request.method == "POST":
        keyword = request.form['keyword']
        page = request.args.get('page', type=int, default=1)  # 페이지
        keyword_list = Question.query.filter(Question.subject.like(f"%{keyword}%"))
        keyword_list = keyword_list.paginate(page=page, per_page=20)
        
        flash(f"{keyword}로 검색한 결과입니다.")
        return render_template("functions/board.html", question_list=keyword_list, keyword=keyword)

    return redirect(url_for("function.board"))
    
@bp_func.route('/detail/<int:question_id>/')
def detail(question_id):
    question = Question.query.get_or_404(question_id)
    return render_template('functions/question_detail.html', question=question)

@bp_func.route("/answer_create/<int:question_id>", methods=('POST',))
def create(question_id):
    question = Question.query.get_or_404(question_id)
    content = request.form['content']
    answer = Answer(content=content, create_date=datetime.now())
    question.answer_set.append(answer)
    db.session.commit()
    
    flash("댓글이 등록되었습니다.")
    return redirect(url_for('function.detail', question_id=question_id))

@bp_func.route("/question_create/", methods=('GET', 'POST'))
def write_question():
    user_id = session.get("user_id")
    if not user_id:
        flash("로그인 후에 이용해주세요.")
        return redirect(url_for("function.board"))
    else:
        if request.method == 'POST':
            username = User.query.filter_by(userid=user_id).first().userid
            subject = request.form['subject']
            content = request.form['content']
            create_date = datetime.now()
            
            q = Question(subject=subject, content=content, create_date=create_date, user_id=username)
            db.session.add(q)
            db.session.commit()
            
            flash("등록 완료 !")
            return redirect(url_for('function.board'))

    return render_template("functions/write_question.html")