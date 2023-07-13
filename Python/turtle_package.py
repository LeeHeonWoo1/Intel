from turtle import *
shape("turtle")
col = ["orange", "limegreen", "gold", "plum", "tomato"]
for i in range(5):
    color(col[i])
    forward(200)
    left(144)
reset()

col = ["red", "green", 'blue', "black", "gray"]
for i in range(5):
    color(col[i])
    circle(100)
    left(72)
done()