import math

import numpy as np

from baba.utils import add_img_text
from baba.rendering import fill_coords, point_in_circle, point_in_rect, point_in_triangle, rotate_fn


properties = [
    'is_stop',
    'is_push',
    'is_goal',
    'is_defeat',
    'is_agent',
    'is_pull',
    'is_move',
    'is_open',
    'is_shut'
]

objects = [
    'fball',
    'fwall',
    'fdoor',
    "fkey",
    'baba'
]

name_mapping = {
    'fwall': 'wall',
    'fball': 'ball',
    'fdoor': 'door',
    'fkey': 'key',
    'is_push': 'push',
    'is_stop': 'stop',
    'is_goal': 'win',
    'is_defeat': 'lose',
    'is': 'is',
    'is_agent': 'you',
    'is_pull': 'pull',
    'is_move': 'move',
    'is_open': 'open',
    'is_shut': 'shut'
}
# by default, add the displayed name is the type of the object
name_mapping.update({o: o for o in objects if o not in name_mapping})
name_mapping_inverted = {v: k for k, v in name_mapping.items()}

TILE_PIXELS = 32

# Map of color names to RGB values
COLORS = {
    "red": np.array([255, 0, 0]),
    "green": np.array([0, 255, 0]),
    "blue": np.array([0, 0, 255]),
    "purple": np.array([112, 39, 195]),
    "yellow": np.array([255, 255, 0]),
    "grey": np.array([100, 100, 100]),
}

COLOR_NAMES = sorted(list(COLORS.keys()))

# Used to map colors to integers
COLOR_TO_IDX = {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5, "white": 6}

IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

# Map of object type to integers
OBJECT_TO_IDX = {
    "unseen": 0,
    "empty": 1,
    "wall": 2,
    "floor": 3,
    "door": 4,
    "key": 5,
    "ball": 6,
    "box": 7,
    "goal": 8,
    "lava": 9,
    "agent": 10,
}

IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

def add_object_types(object_types):
    last_idx = len(OBJECT_TO_IDX)-1
    OBJECT_TO_IDX.update({
        t: last_idx+1+i
    for i, t in enumerate(object_types)})


def add_color_types(color_types):
    last_idx = len(COLOR_TO_IDX)-1
    COLOR_TO_IDX.update({
        t: last_idx+1+i
    for i, t in enumerate(color_types)})


add_color_types(name_mapping.values())
add_object_types(objects)
add_object_types(['rule', 'rule_object', 'rule_is', 'rule_property', 'rule_color'])


def make_obj(name: str, color: str = None):
    """
    Make an object from a string name
    """
    kwargs = {'color': color} if color is not None else {}

    # TODO: make it more general
    if name == "fwall" or name == "wall":
        obj_cls = FWall
    elif name == "fball" or name == "ball":
        obj_cls = FBall
    elif name == "fkey" or name == "key":
        obj_cls = FKey
    elif name == "fdoor" or name == "door":
        obj_cls = FDoor
    elif name == "baba":
        obj_cls = Baba
    else:
        raise ValueError(name)

    return obj_cls(**kwargs)


class WorldObj:
    """
    Base class for grid world objects
    """

    def __init__(self, type, color):
        assert type in OBJECT_TO_IDX, type
        assert color in COLOR_TO_IDX, color
        self.type = type
        self.color = color
        self.contains = None

        # Initial position of the object
        self.init_pos = None

        # Current position of the object
        self.cur_pos = None

    def is_agent(self):
        return False

    def is_move(self):
        return False

    def is_goal(self):
        return False

    def is_defeat(self):
        return False

    def can_overlap(self):
        """Can the agent overlap with this?"""
        return False

    def is_push(self):
        """Can the agent push this?"""
        return False

    def is_pull(self):
        return False

    def can_pickup(self):
        """Can the agent pick this up?"""
        return False

    def can_contain(self):
        """Can this contain another object?"""
        return False

    def see_behind(self):
        """Can the agent see behind this object?"""
        return True

    def toggle(self, env, pos):
        """Method to trigger/toggle an action this object performs"""
        return False

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""
        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], 0)

    @staticmethod
    def decode(type_idx, color_idx, state):
        """Create an object from a 3-tuple state description"""

        obj_type = IDX_TO_OBJECT[type_idx]
        color = IDX_TO_COLOR[color_idx]

        if obj_type == "empty" or obj_type == "unseen":
            return None

        # State, 0: open, 1: closed, 2: locked
        is_open = state == 0
        is_locked = state == 2

        if obj_type == "wall":
            v = Wall(color)
        elif obj_type == "floor":
            v = Floor(color)
        elif obj_type == "ball":
            v = Ball(color)
        elif obj_type == "key":
            v = Key(color)
        elif obj_type == "box":
            v = Box(color)
        elif obj_type == "door":
            v = Door(color, is_open, is_locked)
        elif obj_type == "goal":
            v = Goal()
        elif obj_type == "lava":
            v = Lava()
        else:
            assert False, "unknown object type in decode '%s'" % obj_type

        return v

    def render(self, r):
        """Draw this object with the given renderer"""
        raise NotImplementedError


class Wall(WorldObj):
    def __init__(self, color="grey"):
        super().__init__("wall", color)

    def see_behind(self):
        return False

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class RuleBlock(WorldObj):
    """
    By default, rule blocks can be pushed by the agent.
    """
    def __init__(self, name, type, color, is_push=True):
        super().__init__(type, color)
        self._is_push = is_push
        self.name = name = name_mapping.get(name, name)
        self.margin = 10
        img = np.zeros((96-2*self.margin, 96-2*self.margin, 3), np.uint8)
        add_img_text(img, name)
        self.img = img

    def can_overlap(self):
        return False

    def is_push(self):
        return self._is_push

    def render(self, img):
        fill_coords(img, point_in_rect(0.06, 0.94, 0.06, 0.94), [235, 235, 235])
        img[self.margin:-self.margin, self.margin:-self.margin] = self.img

    # TODO: different encodings of the rule blocks for the agent observation
    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""
        # RuleBlock characterized by their name instead of color
        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.name], 0)


class RuleObject(RuleBlock):
    def __init__(self, obj, is_push=True):
        obj = name_mapping_inverted[obj] if obj not in objects else obj
        # TODO: red push is win (push is a rule_obj but not in objects)
        # assert obj in objects, "{} not in {}".format(obj, objects)

        super().__init__(obj, 'rule_object', 'purple', is_push=is_push)
        self.object = obj


class RuleProperty(RuleBlock):
    def __init__(self, property, is_push=True):
        property = name_mapping_inverted[property] if property not in properties else property
        assert property in properties, "{} not in {}".format(property, properties)

        super().__init__(property, 'rule_property', 'purple', is_push=is_push)
        self.property = property


class RuleIs(RuleBlock):
    def __init__(self, is_push=True):
        super().__init__('is', 'rule_is', 'purple', is_push=is_push)


class RuleColor(RuleBlock):
    def __init__(self, obj_color, is_push=True):
        assert obj_color in COLOR_TO_IDX, "{} not in {}".format(obj_color, COLOR_TO_IDX)

        super().__init__(obj_color, 'rule_color', 'purple', is_push=is_push)
        self.obj_color = obj_color


class Ruleset:
    """
    Each object in the env has a reference to the ruleset object, which is automatically updated (would have to manually
    update it if were using a dict instead).
    """
    def __init__(self, ruleset_dict):
        self.ruleset_dict = ruleset_dict

    def set(self, ruleset_dict):
        self.ruleset_dict = ruleset_dict

    def __getitem__(self, item):
        return self.ruleset_dict[item]

    def __setitem__(self, key, value):
        self.ruleset_dict[key] = value

    def __str__(self):
        return f'Ruleset dict: {self.ruleset_dict}'

    # TODO: cause infinite loop when using vec env
    # def __getattr__(self, item):
    #     return getattr(self.ruleset_dict, item)

    def get(self, *args, **kwargs):
        return self.ruleset_dict.get(*args, **kwargs)



def make_prop_fn(prop: str):
    """
    Make a method that retrieves the property of an instance of FlexibleWorldObj in the ruleset
    """
    def _get_color_set(ruleset, typ, prop):
        return ruleset[prop].get(typ + "_color", [])

    def get_prop(self: FlexibleWorldObj):
        # retrieve the type and color specific to the instance 'self' (the function is the same for all instances)
        typ = self.type
        color = self.color
        ruleset = self.get_ruleset()


        # TODO: cleaner way to implement implicit rules? e.g. is_pull, is_agent implies is_stop
        if prop == 'is_stop':
            if ruleset['is_pull'].get(typ, False) and color in _get_color_set(ruleset, typ, 'is_pull'):
                ruleset['is_stop'][typ] = True
            if ruleset['is_agent'].get(typ, False) and color in _get_color_set(ruleset, typ, 'is_agent'):
                ruleset['is_stop'][typ] = True

        # check for rules specific to a color e.g. {"is_goal": {"fball": True, "fball_color": [0]}}
        color_set = _get_color_set(ruleset, typ, prop)
        if ruleset[prop].get(typ, False):  # object type set to True
            # if no specified color or object fits color specifications
            if (len(color_set) == 0) or (color in color_set):
                return True
        return False

    return get_prop


class FlexibleWorldObj(WorldObj):
    def __init__(self, type, color):
        assert type in objects, "{} not in {}".format(type, objects)
        super().__init__(type, color)
        self.name = name_mapping[type]  # pretty name
        # direction in which the object is facing
        self.dir = 0  # order: right, down, left, up

        for prop in properties:
            # create a method for each property and bind it to the class (same for all the instances of that class)
            setattr(self.__class__, prop, make_prop_fn(prop))

    def set_ruleset(self, ruleset):
        self._ruleset = ruleset

    def get_ruleset(self):
        return self._ruleset

    # compatibility with WorldObj
    def can_overlap(self):
        return not self.is_stop()


class FWall(FlexibleWorldObj):
    def __init__(self, color="grey"):
        super().__init__("fwall", color)

    def render(self, img):
        fill_coords(img, point_in_rect(0.2, 0.8, 0.2, 0.8), COLORS[self.color])


class FBall(FlexibleWorldObj):
    def __init__(self, color="green"):
        super().__init__("fball", color)

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])


class FDoor(FlexibleWorldObj):
    def __init__(self, color="red"):
        super().__init__("fdoor", color)

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""
        # TODO: don't need to encode the state
        state = 0
        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], state)

    def render(self, img):
        c = COLORS[self.color]

        fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
        fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0, 0, 0))
        fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), c)
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0, 0, 0))

        # Draw door handle
        fill_coords(img, point_in_circle(cx=0.75, cy=0.50, r=0.08), c)


class FKey(FlexibleWorldObj):
    def __init__(self, color="blue"):
        super().__init__("fkey", color)

    def render(self, img):
        c = COLORS[self.color]

        # Vertical quad
        fill_coords(img, point_in_rect(0.50, 0.63, 0.31, 0.88), c)

        # Teeth
        fill_coords(img, point_in_rect(0.38, 0.50, 0.59, 0.66), c)
        fill_coords(img, point_in_rect(0.38, 0.50, 0.81, 0.88), c)

        # Ring
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.190), c)
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.064), (0, 0, 0))


class Baba(FlexibleWorldObj):
    def __init__(self, color="white"):
        super().__init__("baba", color)

    def render(self, img):
        tri_fn = point_in_triangle(
            (0.12, 0.19),
            (0.87, 0.50),
            (0.12, 0.81),
        )

        # Rotate the agent based on its direction
        tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * self.dir)
        fill_coords(img, tri_fn, (255, 255, 255))
