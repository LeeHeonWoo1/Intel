from geti import db

class User(db.Model):
    username = db.Column(db.String(20), nullable=False)
    userid = db.Column(db.String(40), primary_key=True)
    userpw = db.Column(db.String(30), nullable=False)
    user_phone = db.Column(db.String(60), nullable=False)

class Question(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    subject = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text(), nullable=False)
    create_date = db.Column(db.DateTime(), nullable=False)
    user_id = db.Column(db.String(200), db.ForeignKey('user.userid', ondelete='CASCADE'), nullable=True, server_default='관리자')
    user = db.relationship('User', backref=db.backref('question_set'))

class Answer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    question_id = db.Column(db.Integer, db.ForeignKey('question.id', ondelete='CASCADE'))
    question = db.relationship('Question', backref=db.backref('answer_set'))
    content = db.Column(db.Text(), nullable=False)
    create_date = db.Column(db.DateTime(), nullable=False)