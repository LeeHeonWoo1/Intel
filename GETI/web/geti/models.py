from geti import db

class User(db.Model):
    username = db.Column(db.String(20), nullable=False)
    userid = db.Column(db.String(40), primary_key=True)
    userpw = db.Column(db.String(30), nullable=False)
    user_phone = db.Column(db.String(60), nullable=False)
    