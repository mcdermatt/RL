import pyglet
from pyglet.gl import *
from pyglet.window import mouse


class Drainage_Window(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_minimum_size(300, 300)

        self.label = None

    def on_mouse_press(self, x, y, button, modifiers):
        if button == mouse.LEFT:
            self.label = pyglet.text.Label('Click!',
                                  font_name='Times New Roman',
                                  font_size=16,
                                  x=x, y=y,
                                  anchor_x='center', anchor_y='center')

    def on_draw(self):
        self.clear()
        if self.label:
            self.label.draw()


if __name__ == '__main__':
    window = Drainage_Window(width=1920 // 4, height=1080//4,
                             caption="Drainage Network", resizable=True)
    glClearColor(0.7, 0.7, 1, 1)

    pyglet.app.run()