import tkinter as tk

# tkinter = GUI 프로그래밍을 도와주는 라이브러리.

root = tk.Tk() # 메인 화면 객체 생성
root.geometry("800x600+800+300") # GUI 창 크기 조정. "200x100"이 의미하는 바는 가로 200, 세로 100의 크기를 의미한다.
lbl = tk.Label(text="Label") # tk.Label = 화면상에 글자를 띄운다. 띄우려는 글자는 객체 내부에 표시된 속성인 text=""라는 속성을 이용해 글자를 지정할 수 있다.
btn = tk.Button(text="Push") # tk.Button = 화면 상에 버튼을 생성한다. Label과 마찬가지로 text라는 속성을 통해 버튼 위에 적혀질 글자를 지정할 수 있다.
lbl.pack() # pack() = 만든 라벨이나 버튼을 화면에 최종적으로 적용하는 과정
btn.pack()

root.mainloop() # 메인 화면 구동