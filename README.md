<div align="center">

<img src="static\logo.png#gh-light-mode-only" width="400">
<img src="static\logo_dark.png#gh-dark-mode-only" width="400">

<h2>Break the Rules to Beat the Benchmark</h2>

</div>

<div align="center">
<img src="static\demo.gif" width="300">
</div>


## Installation
```
git clone https://github.com/nacloos/baba-is-ai
cd baba-is-ai
pip install -e .
```

## Usage
To play an environment, run this command:
```bash
python baba/play.py --env two_room-break_stop-make_win
```
The --env argument specifies the ID of the environment. Once the game opens, use the arrow keys to move the agent.

You can also create a Gym environment object:
```python
import baba

env_id = "two_room-break_stop-make_win"
env = baba.make(f"env/{env_id}")
```

To list all available environment IDs, run this code:
```python
import baba

print(baba.make("env/*").keys())
```

## Citation
If you use this project in your research, please cite:
```

```
