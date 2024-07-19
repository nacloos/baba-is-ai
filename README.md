<div align="center">

<img src="static\logo.png#gh-light-mode-only" width="400">
<img src="static\logo_dark.png#gh-dark-mode-only" width="400">

<h2>Break the Rules to Beat the Benchmark</h2>

</div>

<div align="center">
<img src="static\demo.gif" width="300">
</div>
<div align="center">
 <a href="https://arxiv.org/pdf/2407.13729">Paper</a> 
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
The `--env` argument specifies the ID of the environment. Once the game opens, use the arrow keys to move the agent.

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
@inproceedings{
  cloos2024baba,
  title={Baba Is AI: Break the Rules to Beat the Benchmark},
  author={Nathan Cloos and Meagan Jens and Michelangelo Naim and Yen-Ling Kuo and Ignacio Cases and Andrei Barbu and Christopher J Cueva},
  booktitle={ICML 2024 Workshop on LLMs and Cognition},
  year={2024}
}
```
