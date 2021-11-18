from turtle import *


def virus():
    setheading(90)
    setposition(0, 100)
    speed(50)
    bgcolor('black')
    color('cyan')
    b = 200
    while b > 0:
        left(b)
        print(heading())
        forward(b * 2)
        b -= 1


class Fractal:
    speed(50)
    bgcolor('black')
    color('cyan')
    setheading(90)

    def __init__(self, sequence, amount, angle=30, line_length=50, dimension=2, current_heading=90,
                 current_position=(0, 0)):
        self.sequence = sequence
        self.amount = amount
        self.angle = angle
        self.line_length = line_length
        self.dimension = dimension

        self.actions = {
            'F': self.replace,
            '|': self.draw_line,
            '+': self.turn_right,
            '-': self.turn_left,
            '(': self.save_position,
            ')': self.set_position
        }

        self.current_position = current_position
        self.current_heading = current_heading

    def fractal(self):

        for letter in self.sequence:
            self.actions[letter]()

    def replace(self):

        if self.amount > 1:
            Fractal(self.sequence, self.amount - 1,
                    line_length=int(self.line_length / self.dimension),
                    current_heading=self.current_heading,
                    current_position=self.current_position,
                    angle=self.angle,
                    dimension=self.dimension).fractal()

        else:
            self.draw_line()

    def draw_line(self):
        forward(self.line_length)

    def turn_right(self):
        right(self.angle)

    def turn_left(self):
        left(self.angle)

    def save_position(self):
        self.current_heading = heading()
        self.current_position = position()

    def set_position(self):
        setheading(self.current_heading)
        setposition(self.current_position)


virus()
hideturtle()

# tree = Fractal('|(++F)(-F)', 5, angle=30)
tree = Fractal('F-F++F-F', 4, angle=60, line_length=200, dimension=3, current_heading=0)
penup()
setposition((-300, 0))
setheading(0)
pendown()
tree.fractal()

ts = getscreen()

ts.getcanvas().postscript(file="./duck.ps")
x = 0
