from flask import Blueprint, render_template, url_for, request, flash, session, g
from werkzeug.utils import redirect

from geti import db
from geti.models import Question

bp_func = Blueprint('function', __name__, url_prefix='/func')

@bp_func.route("/board/")
def board():
    question_list = Question.query.order_by(Question.create_date.desc())
    return render_template("functions/board.html", question_list=question_list)

@bp_func.route('/detail/<int:question_id>/')
def detail(question_id):
    question = Question.query.get_or_404(question_id)
    return render_template('functions/question_detail.html', question=question)