# MoonRTX
MoonRTX is ray-traced, ultra-realistic Moon observatory developed in Python. It uses [PlotOptiX](https://github.com/rnd-team-dev/plotoptix) package and is based on PlotOptiX use example [making_the_moon__displacement](https://github.com/rnd-team-dev/plotoptix/blob/master/examples/3_projects/moon/making_the_moon__displacement.ipynb).

## Requirements
- Python
- NVIDIA RTX graphics card

## Install
```bash
git clone https://github.com/albireo77/moonrtx.git  # or download and unpack source code zip file
cd moonrtx
pip install -r requirements.txt
```
## Run (examples)

`python -m moonrtx.main --help`

`python -m moonrtx.main --time "2023-05-28T19:39:00+01:00" --lat 50.1 --lon 20.0`

`python -m moonrtx.main --lat -35.1 --lon -20.4 --downscale 4 --light-intensity 100`

`python -m moonrtx.main --lat 0 --lon 0 --init-view "2025-03-07T18.53.00+01.00_lat+50.000000_lon+20.000000_cambMOZPzVT28IXi6y_bMOZP6uZGsEXi6y_AAAAAAAAAAAAAIA_5oB3QA"`  
where `--init-view` parameter value taken from the screenshot default filename

## Notes
- On first run, around 9GB of data (most of all [Lunar Orbiter Laser Altimeter](https://science.nasa.gov/mission/lro/lola) elevation map) is to be downloaded so internet connection and sufficient disk space is required.
- Rendering Moon scene can take around 1 minute depending on `--downscale` parameter  value.

## Keyboard and Mouse Actions
G - Toggle selenographic grid  
L - Toggle standard labels  
S - Toggle spot labels  
I - Toggle upside down view  
R - Reset view to initial state  
V - Reset view to that based on ephemeris (useful after starting with `--init-view` parameter)  
C - Center view on point under cursor  
F - Search for Moon features (craters, mounts etc.)  
Arrow keys - Navigate view  
F12 - Save image  
Hold and drag left mouse button - Rotate the eye around Moon  
Hold shift + left mouse button and drag up/down - Zoom out/in  
Hold and drag right mouse button - Rotate Moon around the eye  
Hold shift + right mouse button and drag up/down - Move eye backward/forward  

## Screens
![Plato](images/plato.jpg)
![Grid](images/grid.jpg)





