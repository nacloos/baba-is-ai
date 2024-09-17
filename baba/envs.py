from dataclasses import dataclass
from functools import partial
from typing import Iterable
import numpy as np

from baba.grid import BabaIsYouGrid, BabaIsYouEnv, put_rule, place_rule
from baba.play import play
from baba.world_object import FBall, Baba, make_obj, RuleObject, Wall
from baba import make, register



def put_obj(env, obj, pos):
    """
    Put an object at a position in the grid.
    Args:
        env: environment
        obj: object to put, tuple (color, name) or string name
        pos: position to put the object, (i, j)
    """
    if isinstance(obj, tuple):
        color, obj = obj
    else:
        color = None
    obj = make_obj(obj, color=color) if isinstance(obj, str) else obj
    env.put_obj(obj, *pos)


def place_obj(env, obj, top=None, size=None):
    """
    Place an object at a random empty position in the grid.
    Args:
        env: environment
        obj: object to place, tuple (color, name) or string name
        top: top left position of the area to place the object
        size: size of the area to place the object
    """
    if isinstance(obj, tuple):
        color, obj = obj
    else:
        color = None
    obj = make_obj(obj, color=color) if isinstance(obj, str) else obj
    pos = env.place_obj(obj, top=top, size=size)
    return pos


def break_rule(env, rule, new_pos={}, block_idx=None):
    """
    Break a rule by moving one of its blocks to a new position.
    Args:
        rule: rule to break
        new_pos: new position for the block
        block_idx: index of the block to move
    """
    rule_pos = env.init_rules[rule]

    if isinstance(new_pos, dict):
        new_pos = env.place_obj(None, **new_pos)  # pos constraints the new_pos for the rule block

    if block_idx is None:
        # pick one block of the rule and displace it
        block_idx = np.random.choice(len(rule_pos))
    elif isinstance(block_idx, Iterable):
        block_idx = block_idx[np.random.choice(len(block_idx))]

    p = rule_pos[block_idx]
    env.change_obj_pos(p, new_pos)
    # env.solution[rule] = {
    #     "push": (new_pos, p)
    # }


NAMES = ["ball", "key", "door"]
COLORS = ["red", "blue", "green"]
OBJECTS = [
    (color, name) for color in COLORS for name in NAMES
]
ACTIONS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0)
}

@register("env/you_win")
class YouWinEnv(BabaIsYouEnv):
    def __init__(self, width=6, height=6, fixed_you=False, **kwargs):
        self.fixed_you = fixed_you
        super().__init__(width=width, height=height, **kwargs)

    def _gen_grid(self, width, height, params=None):
        # randomly sample the parameters
        indices = np.random.choice(len(OBJECTS), size=4, replace=False)
        win_obj, you_obj, distractor1, distractor2 = [OBJECTS[i] for i in indices]

        self.grid = BabaIsYouGrid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        put_obj(self, Wall(), (1, 3))
        put_obj(self, Wall(), (1, 4))
        put_obj(self, Wall(), (4, 3))
        put_obj(self, Wall(), (4, 4))

        if not self.fixed_you:
            # randomly sample the positions
            all_pos = [(2, 3), (3, 3), (2, 4), (3, 4)]
            you_obj_pos = all_pos.pop(np.random.choice(4))
        else:
            you_obj_pos = (2, 3)
            all_pos = [(2, 4), (3, 3), (3, 4)]

        # make sure the win object is one cell away from the you object
        # e.g. if you object is at (2, 3), win object can be at (2, 4) or (3, 3)
        while True:
            idx = np.random.choice(3)
            win_obj_pos = all_pos[idx]
            if abs(win_obj_pos[0] - you_obj_pos[0]) + abs(win_obj_pos[1] - you_obj_pos[1]) == 1:
                break
        all_pos.pop(idx)
        distractor1_pos = all_pos.pop(np.random.choice(2))
        distractor2_pos = all_pos[0]

        put_rule(self, you_obj, "you", positions=(1, 1))
        put_rule(self, win_obj, "win", positions=(1, 2))
        put_obj(self, you_obj, you_obj_pos)
        put_obj(self, win_obj, win_obj_pos)
        put_obj(self, distractor1, distractor1_pos)
        put_obj(self, distractor2, distractor2_pos)

        # action to move the you object to the win object
        target_action = None
        for action, (dx, dy) in ACTIONS.items():
            if (you_obj_pos[0] + dx, you_obj_pos[1] + dy) == win_obj_pos:
                target_action = action
                break
        self.target_action = target_action

        self.objects = {
            you_obj: you_obj_pos,
            win_obj: win_obj_pos,
            distractor1: distractor1_pos,
            distractor2: distractor2_pos
        }

        self.active_rules = [
            f"{you_obj[0]} {you_obj[1]} is you",
            f"{win_obj[0]} {win_obj[1]} is win"
        ]

        self.target_plan = f"goto[{win_obj}]"


@register("env/you_win-fixed_you")
class YouWinFixedYouEnv(YouWinEnv):
    def __init__(self, **kwargs):
        super().__init__(fixed_you=True, **kwargs)


@register("env/make_win-distr_obj_rule")
class MakeWinEnv(BabaIsYouEnv):
    def __init__(
            self,
            width=8,
            height=8,
            color_in_rule=False,
            break_win_rule=True,
            distractor_obj=True,
            distractor_rule_block=True,
            irrelevant_rule_distractor=False,
            distractor_win_rule=False,
            win_obj_set=None,
            **kwargs
        ):
        self.color_in_rule = color_in_rule
        self.break_win_rule = break_win_rule
        self.distractor_obj = distractor_obj
        self.distractor_rule_block = distractor_rule_block
        self.irrelevant_rule_distractor = irrelevant_rule_distractor
        # add a distractor active win rule (not just a rule block for the distractor object)
        self.distractor_win_rule = distractor_win_rule

        self.win_obj_set = win_obj_set if win_obj_set is not None else NAMES

        super().__init__(width=width, height=height, **kwargs)

    def _sample_objects(self):
            # Sample the objects
        if self.color_in_rule:
            # obj1 and obj2 can be of the same type but different colors
            obj1_idx, obj2_idx, obj3_idx = np.random.choice(len(OBJECTS), size=3, replace=False)
            obj1 = OBJECTS[obj1_idx]
            obj2 = OBJECTS[obj2_idx]
            # obj3 only useful when distractor_rule_block is True and irrelevant_rule_distractor is True
            obj3 = OBJECTS[obj3_idx]
        else:
            # make sure obj1 and obj2 are different object types
            obj1_name, obj2_name, obj3_name = np.random.choice(len(NAMES), size=3, replace=False)
            obj1_color, obj2_color, obj3_color = np.random.choice(len(COLORS), size=3, replace=True)
            obj1 = (COLORS[obj1_color], NAMES[obj1_name])
            obj2 = (COLORS[obj2_color], NAMES[obj2_name])
            obj3 = (COLORS[obj3_color], NAMES[obj3_name])
        return obj1, obj2, obj3

    def _gen_grid(self, width, height, params=None):
        self.grid = BabaIsYouGrid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Sample the objects
        obj1, obj2, obj3 = self._sample_objects()
        # TODO: constraint only object name (don't support color constraint yet)
        while obj1[1] not in self.win_obj_set:
            obj1, obj2, obj3 = self._sample_objects()

        # Add the rules
        put_rule(self, "baba", "you", positions=(1, 6))

        if self.color_in_rule:
            win_obj = obj1
            block_idx = 1
        else:
            win_obj = obj1[1]
            block_idx = 0
        # add win rule
        put_rule(self, win_obj, "win", positions=(1, 1))
        # break the win rule
        if self.break_win_rule:
            break_rule(self, (win_obj, "win"), new_pos={"top": (2, 2), "size": (4, 4)}, block_idx=block_idx)

        if self.distractor_rule_block:
            # add distractor rule block for the other object
            # but place it such that it is impossible to make the obj2 win
            positions = [
                (4, 6), (5, 6), (6, 6),  # bottom row
                (4, 1), (5, 1), (6, 1),  # top row
                (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6)   # right column
            ]
            pos = positions[np.random.choice(len(positions))]

            if self.irrelevant_rule_distractor:
                # add a rule block that is different from obj1 and obj2
                put_obj(self, RuleObject(obj3[1]), pos)
            else:
                put_obj(self, RuleObject(obj2[1]), pos)

        # Place the objects and agent in the grid
        place_obj(self, "baba", top=(1, 2))
        place_obj(self, obj1, top=(1, 2))
        if self.distractor_obj:
            place_obj(self, obj2, top=(1, 2))

        # add extra distractor win rule
        if self.distractor_win_rule:
            put_rule(self, obj3[1], "win", positions=(4, 4))

        if self.color_in_rule:
            self.win_rule = f"{win_obj[0]} {win_obj[1]} is win"
        else:
            self.win_rule = f"{win_obj} is win"
        self.win_obj = win_obj

        if self.break_win_rule:
            self.target_plan = f"make[{self.win_rule}], goto[{self.win_obj}]" if self.color_in_rule else f"make[{self.win_rule}], goto[{self.win_obj}]"
        else:
            self.target_plan = f"goto[{self.win_obj}]"


@register("env/goto_win-distr_obj_rule")
class GotoWinEnv(MakeWinEnv):
    def __init__(self, **kwargs):
        super().__init__(break_win_rule=False, **kwargs)

@register("env/goto_win")
class GotoWinNoDistractorEnv(MakeWinEnv):
    def __init__(self, **kwargs):
        super().__init__(break_win_rule=False, distractor_obj=False, distractor_rule_block=False, **kwargs)

@register("env/goto_win-distr_obj")
class GotoWinNoDistractorEnv(MakeWinEnv):
    def __init__(self, **kwargs):
        super().__init__(break_win_rule=False, distractor_obj=True, distractor_rule_block=False, **kwargs)

@register("env/goto_win-distr_rule")
class GotoWinNoDistractorEnv(MakeWinEnv):
    def __init__(self, **kwargs):
        super().__init__(break_win_rule=False, distractor_obj=False, distractor_rule_block=True, **kwargs)

@register("env/goto_win-distr_obj-irrelevant_rule")
class GotoWinNoDistractorEnv(MakeWinEnv):
    def __init__(self, **kwargs):
        super().__init__(break_win_rule=False, distractor_obj=True, distractor_rule_block=True, irrelevant_rule_distractor=True, **kwargs)


@register("env/goto_win-distr_win_rule")
class GotoWinNoDistractorEnv(MakeWinEnv):
    def __init__(self, **kwargs):
        super().__init__(break_win_rule=False, distractor_obj=True, distractor_win_rule=True, **kwargs)


@register("env/make_win-distr_obj")
class MakeWinNoDistractorRuleEnv(MakeWinEnv):
    def __init__(self, **kwargs):
        super().__init__(distractor_rule_block=False, **kwargs)

@register("env/make_win-distr_rule")
class MakeWinNoDistractorObjEnv(MakeWinEnv):
    def __init__(self, **kwargs):
        super().__init__(distractor_obj=False, **kwargs)

@register("env/make_win")
class MakeWinNoDistractorEnv(MakeWinEnv):
    def __init__(self, **kwargs):
        super().__init__(distractor_obj=False, distractor_rule_block=False, **kwargs)

@register("env/make_win-distr_obj-irrelevant_rule")
class MakeWinIrrelevantDistractorRuleEnv(MakeWinEnv):
    def __init__(self, **kwargs):
        super().__init__(distractor_rule_block=True, irrelevant_rule_distractor=True, **kwargs)


# ===== Single-room make win splits =====
env_ids = [
    "env/make_win",
    "env/make_win-distr_obj",
    "env/make_win-distr_rule",
    "env/make_win-distr_obj-irrelevant_rule",
    "env/make_win-distr_obj_rule",
]
for env_id in env_ids:
    # NAMES minus "ball"
    win_obj_set = [name for name in NAMES if name != "ball"]
    register(
        f"{env_id}#no_ball_win",
        partial(make, env_id, win_obj_set=win_obj_set)
    )
    register(
        f"{env_id}#only_ball_win",
        partial(make, env_id, win_obj_set=["ball"]),
    )



class TwoRoomEnv(BabaIsYouEnv):
    def __init__(
            self,
            width=13,
            height=9,
            baba_pos="left_pushable",
            obj1_pos="anywhere",
            obj2_pos="anywhere",
            break_stop_rule=False,
            break_win_rule=False,
            distractor_obj=True,
            distractor_rule_block=True,
            irrelevant_rule_distractor=False,
            color_in_rule=False,
            distractor_win_rule=False,
            **kwargs
        ):
        self.color_in_rule = color_in_rule
        self.break_stop_rule = break_stop_rule
        self.break_win_rule = break_win_rule
        self.distractor_obj = distractor_obj
        self.distractor_rule_block = distractor_rule_block
        self.irrelevant_rule_distractor = irrelevant_rule_distractor
        self.distractor_win_rule = distractor_win_rule

        pushable_area_size = (width // 2 - 2, height - 6)
        self.positions = {
            # not all left or right space because want it to be possible to push the objects
            "left_pushable": {
                "top": (2, 3),
                "size": pushable_area_size
            },
            "right_pushable": {
                "top": (width // 2 + 1, 3),
                "size": pushable_area_size
            },
            "right_unpushable": [
                # left border
                *[(width-2, 2+i) for i in range(height-4)],
                # bottom border
                *[(width-2-i, height-2) for i in range(width//2-2)],
            ],
            "left_anywhere": {
                "top": (1, 1),
                "size": (width // 2 - 2, height - 2)
            },
            "right_anywhere": {
                # start at y = 2 to prevent having object at the same place as the obj rule block of the win rule for the make_win envs
                "top": (width // 2 + 1, 2),
                "size": (width // 2 - 2, height - 2)
            },
            "anywhere": {
                "top": (1, 2),
                "size": (width - 2, height - 2)
            }
        }
        self.left_pushable = self.positions["left_pushable"]
        self.right_pushable = self.positions["right_pushable"]

        self.baba_pos = self.positions[baba_pos]
        self.obj1_pos = self.positions[obj1_pos]
        self.obj2_pos = self.positions[obj2_pos]

        super().__init__(width=width, height=height, **kwargs)

    def _gen_grid(self, width, height, params=None):
        self.grid = BabaIsYouGrid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Add the vertical wall dividing the two rooms
        self.grid.vert_wall(width // 2, 1, height - 2, obj_type=lambda: make_obj("wall"))

        # Sample the objects
        if self.color_in_rule:
            # obj1 and obj2 can be of the same type but different colors
            obj1_idx, obj2_idx, obj3_idx = np.random.choice(len(OBJECTS), size=3, replace=False)
            obj1 = OBJECTS[obj1_idx]
            obj2 = OBJECTS[obj2_idx]
            # obj3 only useful when distractor_rule_block is True and irrelevant_rule_distractor is True
            obj3 = OBJECTS[obj3_idx]
        else:
            # make sure obj1 and obj2 are different object types
            obj1_name, obj2_name, obj3_name = np.random.choice(len(NAMES), size=3, replace=False)
            obj1_color, obj2_color, obj3_color = np.random.choice(len(COLORS), size=3, replace=True)
            obj1 = (COLORS[obj1_color], NAMES[obj1_name])
            obj2 = (COLORS[obj2_color], NAMES[obj2_name])
            obj3 = (COLORS[obj3_color], NAMES[obj3_name])

        # Add the rules
        # put "baba is you" such that it cannot be changed
        put_rule(self, "baba", "you", positions=(1, height - 2))
        # the agent should be able to break "wall is stop" if needed

        # with this placement, the agent can make "wall is win" and goto "wall"
        # put_rule(self, "wall", "stop", positions=(2, 2))
        # to avoid that we add an unpushable block
        put_rule(self, "wall", "stop", positions=(1, 2))
        # put_obj(self, Wall(), pos=(1, 3))

        if self.color_in_rule:
            # put "obj1 is win" in the corner so that can place a distractor rule block for obj2 such it's impossible to make obj2 win
            put_rule(self, obj1, "win", positions=(width - 4, 1))
        else:
            put_rule(self, obj1[1], "win", positions=(width - 4, 1))

        # Add the distractor rule block
        if self.distractor_rule_block:
            if self.irrelevant_rule_distractor:
                # add a rule block that is different from obj1 and obj2
                place_obj(self, RuleObject(obj3[1]), **self.positions["right_pushable"])
            elif self.distractor_obj:                
                # add distractor rule block for the other object
                # but place it such that it is impossible to make the obj2 win
                positions = self.positions["right_unpushable"]
                pos = positions[np.random.choice(len(positions))]
                put_obj(self, RuleObject(obj2[1]), pos)
            else:
                # don't need to be unpushable because obj2 is not in the env
                place_obj(self, RuleObject(obj2[1]), **self.positions["right_pushable"])

        # Break the rules if needed
        if self.break_stop_rule:
            break_rule(self, ("wall", "stop"), new_pos=self.left_pushable, block_idx=[1, 2])

        if self.break_win_rule:
            win_obj = obj1 if self.color_in_rule else obj1[1]
            block_idx = 1 if self.color_in_rule else 0
            break_rule(self, (win_obj, "win"), new_pos=self.right_pushable, block_idx=block_idx)

        # Place the objects and agent in the rooms
        baba_pos = place_obj(self, "baba", **self.baba_pos)
        obj1_pos = place_obj(self, obj1, **self.obj1_pos)
        if self.distractor_obj:
            place_obj(self, obj2, **self.obj2_pos)

        # add extra distractor win rule
        if self.distractor_win_rule:
            put_rule(self, obj3[1], "win", positions=(width-4, 4))

        # if win rule active and win object on the left, plan = goto win object
        if not self.break_win_rule and obj1_pos[0] < width // 2:
            self.target_plan = f"goto[{obj1[1]}]"
        # if win rule active, win obj on the right and stop rule inactive, plan = goto win object
        elif not self.break_win_rule and obj1_pos[0] > width // 2 and self.break_stop_rule:
            self.target_plan = f"goto[{obj1[1]}]"
        # if win rule active, win obj on the right and stop rule active, plan = break stop rule, goto win object
        elif not self.break_win_rule and obj1_pos[0] > width // 2 and not self.break_stop_rule:
            self.target_plan = f"break[wall is stop], goto[{obj1[1]}]"
        # if win rule inactive and stop rule inactive, plan = make win rule, goto win obj
        elif self.break_win_rule and self.break_stop_rule:
            self.target_plan = f"make[{obj1[1]} is win], goto[{obj1[1]}]"
        # if win rule inactive and stop rule active, plan = break stop rule, make win rule, goto win obj
        elif self.break_win_rule and not self.break_stop_rule:
            self.target_plan = f"break[wall is stop], make[{obj1[1]} is win], goto[{obj1[1]}]"
        else:
            breakpoint()
            raise ValueError("Invalid configuration")


@register("env/two_room-goto_win")
class TwoRoomGotoWinEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(obj1_pos="left_anywhere", obj2_pos="left_anywhere", distractor_obj=False, distractor_rule_block=False, break_stop_rule=True, **kwargs)


@register("env/two_room-goto_win-distr_obj_rule")
class TwoRoomGotoWinEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(obj1_pos="left_anywhere", obj2_pos="left_anywhere", break_stop_rule=True, **kwargs)


@register("env/two_room-goto_win-distr_rule")
class TwoRoomGotoWinNoDistractorObjEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(obj1_pos="left_anywhere", obj2_pos="left_anywhere", distractor_obj=False, break_stop_rule=True, **kwargs)


@register("env/two_room-goto_win-distr_obj")
class TwoRoomGotoWinNoDistractorObjEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(obj1_pos="left_anywhere", obj2_pos="left_anywhere", distractor_obj=True, distractor_rule_block=False, break_stop_rule=True, **kwargs)


@register("env/two_room-goto_win-distr_obj-irrelevant_rule")
class TwoRoomGotoWinNoDistractorObjEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(obj1_pos="left_anywhere", obj2_pos="left_anywhere", distractor_obj=True, distractor_rule_block=True, irrelevant_rule_distractor=True, break_stop_rule=True, **kwargs)


@register("env/two_room-goto_win-distr_win_rule")
class TwoRoomGotoWinDistrWinRule(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(obj1_pos="left_anywhere", obj2_pos="left_anywhere", distractor_obj=True, distractor_win_rule=True, break_stop_rule=True, **kwargs)


# ===== variants of break stop, goto win =====
@register("env/two_room-break_stop-goto_win-distr_obj_rule")
class TwoRoomBreakStopGotoWinEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=False, obj1_pos="right_anywhere", obj2_pos="right_anywhere", **kwargs)


@register("env/two_room-break_stop-goto_win-distr_obj")
class TwoRoomBreakStopGotoWinEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=False, obj1_pos="right_anywhere", obj2_pos="right_anywhere", distractor_rule_block=False, **kwargs)


@register("env/two_room-break_stop-goto_win-distr_rule")
class TwoRoomBreakStopGotoWinEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=False, obj1_pos="right_anywhere", obj2_pos="right_anywhere", distractor_obj=False, **kwargs)

@register("env/two_room-break_stop-goto_win-distr_obj-irrelevant_rule")
class TwoRoomBreakStopGotoWinEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=False, obj1_pos="right_anywhere", obj2_pos="right_anywhere", distractor_obj=True, distractor_rule_block=True, irrelevant_rule_distractor=True, **kwargs)


@register("env/two_room-break_stop-goto_win")
class TwoRoomBreakStopGotoWinEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=False, obj1_pos="right_anywhere", obj2_pos="right_anywhere", distractor_obj=False, distractor_rule_block=False, **kwargs)


@register("env/two_room-maybe_break_stop-goto_win-distr_obj_rule")
class TwoRoomMaybeBreakStopGotoWinDistrObjRuleEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=False, obj1_pos="anywhere", obj2_pos="anywhere", **kwargs)


@register("env/two_room-maybe_break_stop-goto_win")
class TwoRoomMaybeBreakStopGotoWinEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=False, obj1_pos="anywhere", obj2_pos="anywhere", distractor_obj=False, distractor_rule_block=False, **kwargs)


@register("env/two_room-maybe_break_stop-goto_win-distr_obj")
class TwoRoomMaybeBreakStopGotoWinEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=False, obj1_pos="anywhere", obj2_pos="anywhere", distractor_obj=True, distractor_rule_block=False, **kwargs)


@register("env/two_room-maybe_break_stop-goto_win-distr_rule")
class TwoRoomMaybeBreakStopGotoWinEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=False, obj1_pos="anywhere", obj2_pos="anywhere", distractor_obj=False, distractor_rule_block=True, **kwargs)


@register("env/two_room-maybe_break_stop-goto_win-distr_obj-irrelevant_rule")
class TwoRoomMaybeBreakStopGotoWinEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=False, obj1_pos="anywhere", obj2_pos="anywhere", distractor_obj=True, distractor_rule_block=True, irrelevant_rule_distractor=True, **kwargs)


# ===== variants of make win =====
@register("env/two_room-make_win-distr_obj_rule")
class TwoRoomMakeWinEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=True, break_win_rule=True, obj1_pos="left_anywhere", obj2_pos="left_anywhere", **kwargs)


@register("env/two_room-make_win-distr_rule")
class TwoRoomMakeWinNoDistractorObjEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=True, break_win_rule=True, obj1_pos="left_anywhere", obj2_pos="left_anywhere", distractor_obj=False, **kwargs)


@register("env/two_room-make_win")
class TwoRoomMakeWinNoDistractorEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=True, break_win_rule=True, obj1_pos="left_anywhere", obj2_pos="left_anywhere", distractor_obj=False, distractor_rule_block=False, **kwargs)


@register("env/two_room-make_win-distr_obj-irrelevant_rule")
class TwoRoomMakeWinIrrelevantDistractorRuleEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=True, break_win_rule=True, obj1_pos="left_anywhere", obj2_pos="left_anywhere", distractor_rule_block=True, irrelevant_rule_distractor=True, **kwargs)


@register("env/two_room-make_win-distr_obj")
class TwoRoomMakeWinNoDistractorRuleEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=True, break_win_rule=True, obj1_pos="left_anywhere", obj2_pos="left_anywhere", distractor_rule_block=False, **kwargs)


@register("env/two_room-make_win-distr_win_rule")
class TwoRoomGotoWinDistrWinRule(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(obj1_pos="left_anywhere", obj2_pos="left_anywhere", distractor_obj=True, distractor_win_rule=True, break_win_rule=True, break_stop_rule=True, **kwargs)


# ===== variants of break stop, make win =====
@register("env/two_room-break_stop-make_win-distr_obj_rule")
class TwoRoomBreakStopGotoWinEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=False, break_win_rule=True, obj1_pos="right_anywhere", obj2_pos="right_anywhere", **kwargs)


@register("env/two_room-break_stop-make_win-distr_rule")
class TwoRoomBreakStopMakeWinNoDistractorObjEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=False, break_win_rule=True, obj1_pos="right_anywhere", obj2_pos="right_anywhere", distractor_obj=False, **kwargs)


@register("env/two_room-break_stop-make_win")
class TwoRoomBreakStopMakeWinNoDistractorEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=False, break_win_rule=True, obj1_pos="right_anywhere", obj2_pos="right_anywhere", distractor_obj=False, distractor_rule_block=False, **kwargs)


@register("env/two_room-break_stop-make_win-distr_obj-irrelevant_rule")
class TwoRoomBreakStopMakeWinIrrelevantDistractorRuleEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=False, break_win_rule=True, obj1_pos="right_anywhere", obj2_pos="right_anywhere", distractor_rule_block=True, irrelevant_rule_distractor=True, **kwargs)


@register("env/two_room-break_stop-make_win-distr_obj")
class TwoRoomBreakStopMakeWinNoDistractorRuleEnv(TwoRoomEnv):
    def __init__(self, **kwargs):
        super().__init__(break_stop_rule=False, break_win_rule=True, obj1_pos="right_anywhere", obj2_pos="right_anywhere", distractor_rule_block=False, **kwargs)


@register("env/two_room-make_you")
class TwoRoomMakeYouEnv(BabaIsYouEnv):
    def __init__(
            self,
            width=13,
            height=9,
            baba_pos="left_pushable",
            obj1_pos="right_anywhere",
            obj2_pos="right_anywhere",
            break_stop_rule=False,
            break_win_rule=False,
            distractor_obj=True,
            distractor_rule_block=True,
            irrelevant_rule_distractor=False,
            color_in_rule=False,
            **kwargs
        ):
        self.color_in_rule = color_in_rule
        self.break_stop_rule = break_stop_rule
        self.break_win_rule = break_win_rule
        self.distractor_obj = distractor_obj
        self.distractor_rule_block = distractor_rule_block
        self.irrelevant_rule_distractor = irrelevant_rule_distractor

        pushable_area_size = (width // 2 - 3, height - 6)
        self.positions = {
            # not all left or right space because want it to be possible to push the objects
            "left_pushable": {
                "top": (2, 3),
                "size": pushable_area_size
            },
            "right_pushable": {
                "top": (width // 2 + 2, 3),
                "size": pushable_area_size
            },
            "right_unpushable": [
                # left border
                *[(width-2, 2+i) for i in range(height-4)],
                # bottom border
                *[(width-2-i, height-2) for i in range(width//2-2)],
            ],
            "left_anywhere": {
                "top": (1, 1),
                "size": (width // 2 - 2, height - 2)
            },
            "right_anywhere": {
                # start at y = 2 to prevent having object at the same place as the obj rule block of the win rule for the make_win envs
                "top": (width // 2 + 1, 2),
                "size": (width // 2 - 2, height - 2)
            },
            "anywhere": {
                "top": (1, 2),
                "size": (width - 2, height - 2)
            }
        }
        self.left_pushable = self.positions["left_pushable"]
        self.right_pushable = self.positions["right_pushable"]

        self.baba_pos = self.positions[baba_pos]
        self.obj1_pos = self.positions[obj1_pos]
        self.obj2_pos = self.positions[obj2_pos]

        super().__init__(width=width, height=height, **kwargs)

    def _gen_grid(self, width, height, params=None):
        self.grid = BabaIsYouGrid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Add the vertical wall dividing the two rooms
        self.grid.vert_wall(width // 2, 1, height - 2, obj_type=lambda: make_obj("wall"))

        # Sample the objects
        if self.color_in_rule:
            # obj1 and obj2 can be of the same type but different colors
            obj1_idx, obj2_idx, obj3_idx = np.random.choice(len(OBJECTS), size=3, replace=False)
            obj1 = OBJECTS[obj1_idx]
            obj2 = OBJECTS[obj2_idx]
            # obj3 only useful when distractor_rule_block is True and irrelevant_rule_distractor is True
            obj3 = OBJECTS[obj3_idx]
        else:
            # make sure obj1 and obj2 are different object types
            obj1_name, obj2_name, obj3_name = np.random.choice(len(NAMES), size=3, replace=False)
            obj1_color, obj2_color, obj3_color = np.random.choice(len(COLORS), size=3, replace=True)
            obj1 = (COLORS[obj1_color], NAMES[obj1_name])
            obj2 = (COLORS[obj2_color], NAMES[obj2_name])
            obj3 = (COLORS[obj3_color], NAMES[obj3_name])

        # Add the rules
        # put "baba is you" such that it cannot be changed
        # put_rule(self, "baba", "you", positions=(1, height - 2))
        put_rule(self, "baba", "you", positions=(1, height - 3))
        # the agent should be able to break "wall is stop" if needed

        # with this placement, the agent can make "wall is win" and goto "wall"
        # put_rule(self, "wall", "stop", positions=(2, 2))
        # to avoid that we add an unpushable block
        put_rule(self, "wall", "stop", positions=(1, 1))

        if self.color_in_rule:
            # put "obj1 is win" in the corner so that can place a distractor rule block for obj2 such it's impossible to make obj2 win
            put_rule(self, obj1, "win", positions=(width - 4, 1))
        else:
            put_rule(self, obj1[1], "win", positions=(width - 4, 1))

        # Add the distractor rule block
        if self.distractor_rule_block:
            if self.irrelevant_rule_distractor:
                # add a rule block that is different from obj1 and obj2
                place_obj(self, RuleObject(obj3[1]), **self.positions["left_pushable"])
            elif self.distractor_obj:                
                place_obj(self, RuleObject(obj2[1]), **self.positions["left_pushable"])
            else:
                # don't need to be unpushable because obj2 is not in the env
                place_obj(self, RuleObject(obj2[1]), **self.positions["left_pushable"])

        # Break the rules if needed
        if self.break_stop_rule:
            break_rule(self, ("wall", "stop"), new_pos=self.left_pushable, block_idx=[1, 2])

        if self.break_win_rule:
            win_obj = obj1 if self.color_in_rule else obj1[1]
            block_idx = 1 if self.color_in_rule else 0
            break_rule(self, (win_obj, "win"), new_pos=self.right_pushable, block_idx=block_idx)

        # Place the objects and agent in the rooms
        place_obj(self, "baba", **self.baba_pos)
        place_obj(self, obj1, **self.obj1_pos)
        if self.distractor_obj:
            place_obj(self, obj2, **self.obj2_pos)

        if self.break_win_rule:
            self.target_plan = f"make[{obj2[1]} is you], make[{obj1[1]} is win], goto[{obj1[1]}]"
        else:
            self.target_plan = f"make[{obj2[1]} is you], goto[{obj1[1]}]"


@register("env/two_room-make_you-make_win")
class TwoRoomMakeWinMakeWinEnv(TwoRoomMakeYouEnv):
    def __init__(self, **kwargs):
        super().__init__(break_win_rule=True, **kwargs)


@register("env/two_room-make_wall_win")
class TwoRoomMakeWallWinEnv(BabaIsYouEnv):
    def __init__(
            self,
            width=13,
            height=9,
            baba_pos="left_pushable",
            obj1_pos="right_anywhere",
            obj2_pos="right_anywhere",
            break_stop_rule=False,
            distractor_obj=True,
            distractor_rule_block=True,
            irrelevant_rule_distractor=True,
            **kwargs
        ):
        self.break_stop_rule = break_stop_rule
        self.distractor_obj = distractor_obj
        self.distractor_rule_block = distractor_rule_block
        self.irrelevant_rule_distractor = irrelevant_rule_distractor

        pushable_area_size = (width // 2 - 3, height - 6)
        self.positions = {
            # not all left or right space because want it to be possible to push the objects
            "left_pushable": {
                "top": (2, 3),
                "size": pushable_area_size
            },
            "right_pushable": {
                "top": (width // 2 + 2, 3),
                "size": pushable_area_size
            },
            "right_unpushable": [
                # left border
                *[(width-2, 2+i) for i in range(height-4)],
                # bottom border
                *[(width-2-i, height-2) for i in range(width//2-2)],
            ],
            "left_anywhere": {
                "top": (1, 1),
                "size": (width // 2 - 2, height - 2)
            },
            "right_anywhere": {
                # start at y = 2 to prevent having object at the same place as the obj rule block of the win rule for the make_win envs
                "top": (width // 2 + 1, 2),
                "size": (width // 2 - 2, height - 2)
            },
            "anywhere": {
                "top": (1, 2),
                "size": (width - 2, height - 2)
            }
        }
        self.left_pushable = self.positions["left_pushable"]
        self.right_pushable = self.positions["right_pushable"]

        self.baba_pos = self.positions[baba_pos]
        self.obj1_pos = self.positions[obj1_pos]
        self.obj2_pos = self.positions[obj2_pos]

        super().__init__(width=width, height=height, **kwargs)

    def _gen_grid(self, width, height, params=None):
        self.grid = BabaIsYouGrid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Add the vertical wall dividing the two rooms
        self.grid.vert_wall(width // 2, 1, height - 2, obj_type=lambda: make_obj("wall"))

        # Sample the objects
        obj1_name, obj2_name, obj3_name = np.random.choice(len(NAMES), size=3, replace=False)
        obj1_color, obj2_color, obj3_color = np.random.choice(len(COLORS), size=3, replace=True)
        obj1 = (COLORS[obj1_color], NAMES[obj1_name])
        obj2 = (COLORS[obj2_color], NAMES[obj2_name])
        obj3 = (COLORS[obj3_color], NAMES[obj3_name])

        put_rule(self, "baba", "you", positions=(1, height - 2))
        put_rule(self, "wall", "stop", positions=(2, 2))
        put_rule(self, obj1[1], "win", positions=(width - 4, 1))

        if self.distractor_rule_block:
            if self.irrelevant_rule_distractor:
                # add a rule block that is different from obj1 and obj2
                place_obj(self, RuleObject(obj3[1]), **self.positions["right_pushable"])
            elif self.distractor_obj:
                # add distractor rule block for the other object
                # but place it such that it is impossible to make the obj2 win
                positions = self.positions["right_unpushable"]
                pos = positions[np.random.choice(len(positions))]
                put_obj(self, RuleObject(obj2[1]), pos)
            else:
                # don't need to be unpushable because obj2 is not in the env
                place_obj(self, RuleObject(obj2[1]), **self.positions["right_pushable"])

        # Break the rules if needed
        if self.break_stop_rule:
            break_rule(self, ("wall", "stop"), new_pos=self.left_pushable, block_idx=[1, 2])

        new_pos = self.positions["right_unpushable"][np.random.choice(len(self.positions["right_unpushable"]))]
        break_rule(self, (obj1[1], "win"), new_pos=new_pos, block_idx=0)

        # Place the objects and agent in the rooms
        place_obj(self, "baba", **self.baba_pos)
        place_obj(self, obj1, **self.obj1_pos)
        if self.distractor_obj:
            place_obj(self, obj2, **self.obj2_pos)

        self.target_plan = f"break[wall is stop], make[wall is win], goto[wall]"


if __name__ == "__main__":
    # env = make("env/goto_win-no_distractor")
    # env = make("env/make_win-no_distractor_rule")
    # env = make("env/make_win-no_distractor_obj")
    # env = make("env/make_win-no_distractor")
    # env = make("env/make_win-irrelevant_distractor_rule")

    # env = make("env/two_room-goto_win")
    # env = make("env/two_room-make_win")
    # env = make("env/two_room-make_win-no_distractor_obj")
    # env = make("env/two_room-maybe_break_stop-goto_win")

    # env = make("env/two_room-break_stop-make_win-no_distractor_obj")

    # env = make("env/make_win-distr_obj_rule")
    # env = make("env/make_win-distr_obj_rule#no_ball_win")
    # env = make("env/make_win-distr_obj_rule#only_ball_win")

    # env = make("env/two_room-break_stop-make_win-distr_obj-irrelevant_rule")
    # env = make("env/two_room-make_you-make_win")
    env = make("env/two_room-make_wall_win")

    env.reset()
    obs = env.render(mode="matrix")
    print(obs)

    play(env)
