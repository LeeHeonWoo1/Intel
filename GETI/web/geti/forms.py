from flask_wtf import FlaskForm
from datetime import datetime
from wtforms import StringField, PasswordField
from wtforms.validators import DataRequired, Length, EqualTo

class UserCreateForm(FlaskForm):
    username = StringField('사용자이름', validators=[DataRequired(), Length(min=3, max=25)])
    userid = StringField("사용자ID", validators=[DataRequired(), Length(min=3, max=25)])
    password1 = PasswordField('비밀번호', validators=[
        DataRequired(), 
        EqualTo('password2', '비밀번호가 일치하지 않습니다')
        ]
    )
    password2 = PasswordField('비밀번호확인', validators=[DataRequired()])
    phone = StringField('전화번호', validators=[DataRequired()])
    
class UserLoginForm(FlaskForm):
    userid = StringField('사용자이름', validators=[DataRequired(), Length(min=3, max=25)])
    password = PasswordField('비밀번호', validators=[DataRequired()])
    
class NewQuestionForm(FlaskForm):
    subject = StringField("제목", validators=[DataRequired()])
    content = StringField("본문", validators=[DataRequired()])