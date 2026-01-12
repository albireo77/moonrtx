# MoonRTX
MoonRTX is a ray traced Moon observatory developed in Python. It uses [PlotOptiX](https://github.com/rnd-team-dev/plotoptix) package and is based on PlotOptiX example use [making_the_moon__displacement](https://github.com/rnd-team-dev/plotoptix/blob/master/examples/3_projects/moon/making_the_moon__displacement.ipynb).

## Requirements
- Python
- NVIDIA RTX graphics card

## Installation
```bash
git clone <repo_url>
cd moonrtx
pip install -r requirements.txt
```
Run examples: `python -m moonrtx.main --help` or `python -m moonrtx.main --time "2023-05-28T19:39:00+01:00" --lat 50.0 --lon 20.0`

## Notes
- On first run around 9GB of data must be downloaded.
- Rendering Moon scene can take around 1 minute depending on --downscale parameter  value.
- Available keyboard shortcuts and mouse actions are displayed in program console output.

## Screens
![Plato](images/plato.jpg)
![Grid](images/grid.jpg)





