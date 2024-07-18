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
Run this command to play an enviromnent.
```
python baba/play.py --env two_room-break_stop-make_win
```
The env argument specifies the id of the environment. Use the arrow keys to move the agent.

You can also create a gym enviromnent object.
```python
import baba

env_id = "two_room-break_stop-make_win"
env = baba.make(f"env/{env_id}")
```

List all available environment ids.
```python
baba.make("env/*").keys()
```

## Citation

```

```
