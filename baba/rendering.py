import math

import numpy as np
import matplotlib.pyplot as plt


def downsample(img, factor):
    """
    Downsample an image along both dimensions by some factor
    """

    assert img.shape[0] % factor == 0
    assert img.shape[1] % factor == 0

    img = img.reshape(
        [img.shape[0] // factor, factor, img.shape[1] // factor, factor, 3]
    )
    img = img.mean(axis=3)
    img = img.mean(axis=1)

    return img


def fill_coords(img, fn, color):
    """
    Fill pixels of an image with coordinates matching a filter function
    """

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            yf = (y + 0.5) / img.shape[0]
            xf = (x + 0.5) / img.shape[1]
            if fn(xf, yf):
                img[y, x] = color

    return img


def rotate_fn(fin, cx, cy, theta):
    def fout(x, y):
        x = x - cx
        y = y - cy

        x2 = cx + x * math.cos(-theta) - y * math.sin(-theta)
        y2 = cy + y * math.cos(-theta) + x * math.sin(-theta)

        return fin(x2, y2)

    return fout


def point_in_line(x0, y0, x1, y1, r):
    p0 = np.array([x0, y0])
    p1 = np.array([x1, y1])
    dir = p1 - p0
    dist = np.linalg.norm(dir)
    dir = dir / dist

    xmin = min(x0, x1) - r
    xmax = max(x0, x1) + r
    ymin = min(y0, y1) - r
    ymax = max(y0, y1) + r

    def fn(x, y):
        # Fast, early escape test
        if x < xmin or x > xmax or y < ymin or y > ymax:
            return False

        q = np.array([x, y])
        pq = q - p0

        # Closest point on line
        a = np.dot(pq, dir)
        a = np.clip(a, 0, dist)
        p = p0 + a * dir

        dist_to_line = np.linalg.norm(q - p)
        return dist_to_line <= r

    return fn


def point_in_circle(cx, cy, r):
    def fn(x, y):
        return (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r

    return fn


def point_in_rect(xmin, xmax, ymin, ymax):
    def fn(x, y):
        return x >= xmin and x <= xmax and y >= ymin and y <= ymax

    return fn


def point_in_triangle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    def fn(x, y):
        v0 = c - a
        v1 = b - a
        v2 = np.array((x, y)) - a

        # Compute dot products
        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        # Compute barycentric coordinates
        inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom

        # Check if point is in triangle
        return (u >= 0) and (v >= 0) and (u + v) < 1

    return fn


def highlight_img(img, color=(255, 255, 255), alpha=0.30):
    """
    Add highlighting to an image
    """

    blend_img = img + alpha * (np.array(color, dtype=np.uint8) - img)
    blend_img = blend_img.clip(0, 255).astype(np.uint8)
    img[:, :, :] = blend_img


class Window:
    """
    Window to draw a gridworld instance using Matplotlib
    """

    def __init__(self, title):
        self.no_image_shown = True

        # Create the figure and axes
        self.fig, self.ax = plt.subplots()

        # Show the env name in the window title
        self.fig.canvas.manager.set_window_title(title)

        # Turn off x/y axis numbering/ticks
        self.ax.xaxis.set_ticks_position("none")
        self.ax.yaxis.set_ticks_position("none")
        _ = self.ax.set_xticklabels([])
        _ = self.ax.set_yticklabels([])

        # Flag indicating the window was closed
        self.closed = False

        def close_handler(evt):
            self.closed = True

        self.fig.canvas.mpl_connect("close_event", close_handler)

    def show_img(self, img):
        """
        Show an image or update the image being shown
        """

        # If no image has been shown yet,
        # show the first image of the environment
        if self.no_image_shown:
            self.imshow_obj = self.ax.imshow(img, interpolation="bilinear")
            self.no_image_shown = False
        # Update the image data
        self.imshow_obj.set_data(img)

        # Request the window be redrawn
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        # Let matplotlib process UI events
        plt.pause(0.001)

    def set_caption(self, text):
        """
        Set/update the caption text below the image
        """

        plt.xlabel(text)

    def reg_key_handler(self, key_handler):
        """
        Register a keyboard event handler
        """

        # Keyboard handler
        self.fig.canvas.mpl_connect("key_press_event", key_handler)

    def show(self, block=True):
        """
        Show the window, and start an event loop
        """

        # If not blocking, trigger interactive mode
        if not block:
            plt.ion()

        # Show the plot
        # In non-interative mode, this enters the matplotlib event loop
        # In interactive mode, this call does not block
        plt.show()

    def close(self):
        """
        Close the window
        """

        plt.close()
        self.closed = True
