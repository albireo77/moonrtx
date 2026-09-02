# MoonRTX
MoonRTX is ray-traced, ultra-realistic Moon observatory developed in Python. It uses [PlotOptiX](https://github.com/rnd-team-dev/plotoptix) package and is based on PlotOptiX use example [making_the_moon__displacement](https://github.com/rnd-team-dev/plotoptix/blob/master/examples/3_projects/moon/making_the_moon__displacement.ipynb).

## Requirements
- Python
- pip (Python package manager)
- **NVIDIA RTX graphics card with installed latest NVIDIA drivers (R590 or newer)**
- As of now application was tested only on Windows. I have no possibility to test it under Linux. Nevertheless, it should run on Linux when PlotOptiX prerequisites for this OS are met. More details [here](https://github.com/rnd-team-dev/plotoptix?tab=readme-ov-file#linux-prerequisites)

## Install
```bash
git clone https://github.com/albireo77/moonrtx.git  # or download and unpack source code zip file
cd moonrtx
pip install -r requirements.txt
```
## Run

MoonRTX can be run in 2 ways:
- Command line (examples):

`python -m moonrtx.main --help`

`python -m moonrtx.main --time "2023-05-28T19:39:00" --lat 50.1 --lon 20.0`

`python -m moonrtx.main --time "2023-05-28T19:39:00" --timezone "America/New_York" --lat 40.7 --lon -74.0`  
where `--time` is a local wall clock and `--timezone` names the IANA zone it is read in, defaulting to your computer's own. The zone carries the daylight saving and historical rules, so every date gets the offset that really applied on it - including dates in another country, whose changeover days differ from yours.

`python -m moonrtx.main --lat -35.1 --lon -20.4 --downscale 4 --brightness 100`

`python -m moonrtx.main --lat 50.1 --lon 20.0 --color-file lroc_color_poles.tif --color-downscale 2`  
where `--color-downscale` (1, 2, 4 or 8) decodes the color map at that fraction of its size. Memory needed to load it falls with the square of the factor, so raise it if a large map fails to load.

`python -m moonrtx.main --init-view "2025-03-07T19.53.00+01.00_lat+50.000000_lon+20.000000_viewSNEW_par1_camAAAAAAAAyMIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIA_F6dJQA"`  
where `--init-view` parameter value is taken from the screenshot default filename
- GUI launcher:

`python -m moonrtx.main_gui_launcher`

## Notes
- On first run, around 9GB of data (most of all [Lunar Orbiter Laser Altimeter](https://science.nasa.gov/mission/lro/lola) elevation map) is to be downloaded so internet connection and sufficient disk space is required.
- On first Moon ephemeris calculation, MoonRTX may download a small Skyfield data set into `moonrtx/data/skyfield`: the JPL `de421.bsp` ephemeris (about 16 MB) plus three small Moon orientation kernels used for the direct lunar body-frame rotation.
- The bundled Skyfield Moon orientation kernels support dates from `1900-01-01T00:00:00+00:00` through `2051-01-01T00:00:00+00:00`.
- Local times are handled through IANA timezones (the `tzdata` package), so daylight saving is followed on every date the kernels cover - including the years before it existed and the local mean time in use before standard time.
- Current MoonRTX will work only with PlotOptiX [0.19.2](https://github.com/rnd-team-dev/plotoptix/releases#release-v0.19.2)+
- Downscaling of elevation map can take even around 1 minute depending on `--downscale` parameter value (lower value = more time). Once downscaled for the first time, file is cached as .npy file in `data` directory for future use.
- Color maps beyond a few hundred megapixels can need more RAM than the machine has. `--color-downscale` avoids that: the map is decoded straight at a fraction of its size, so nothing full-size is ever held. As with the elevation map, the result is cached beside the source as an .npy file, and once it is there the source can be deleted to reclaim its space.
- If you don't like default Moon colors in MoonRTX, you can download color map file (e.g. [lroc_color_poles.tif](https://svs.gsfc.nasa.gov/vis/a000000/a004700/a004720/lroc_color_poles.tif) from [NASA SVS CGI Moon Kit](https://svs.gsfc.nasa.gov/4720) with more realistic, though bland, colors) and load it with `--color-file` parameter on program start.

## Keyboard and Mouse Actions
**F1** - Help  
**F2** - Toggle Moon ephemeris panel  
**F3** - Set up the eyepiece / camera field of view frame  
**F4** - Toggle parallactic mode (maintains Moon aligned to celestial north)  
**F5-F8** - Change view orientation (NSWE, NSEW, SNEW, SNWE)  
**F9** - Set time to now (in the session timezone)  
**F10** - Set time to now + start auto-advance  
**F11** - Export time-lapse video (MP4)  
**F12** - Save image  
**1-9** - Create/Remove pin (when pins are ON)  
**G** - Toggle selenographic grid  
**L** - Toggle standard labels  
**S** - Toggle spot labels  
**P** - Toggle pins ON/OFF  
**Y** - Toggle markers for sub-solar and sub-Earth points  
**B** - Toggle the field of view frame (set it up with F3)  
**R** - Reset camera and time to initial state  
**V** - Reset camera to default state (useful after starting with `--init-view` parameter)  
**C** - Center and fix view on point under cursor  
**F** - Search for Moon features (craters, mounts etc.)  
**X** - Find clair-obscur events (Lunar X, Lunar V, Jewelled Handle, Eyes of Clavius, Rupes Recta)  
**U** - Chart when the Moon is up over the coming month  
**K** - Open observation planner (terminator / libration) for Moon feature in status bar  
**I** - Open USGS web page for Moon feature in status bar  
**O** - Open user defined web page (Wikipedia by default) for Moon feature in status bar  
**T** - Open date/time window  
**A/Z** - Increase/Decrease brightness  
**E/D** - Increase/Decrease gamma correction  
**H/J** - Roll view around current view direction  
**Q/W** - Go back/forward in time by step minutes (hold the key to get an animation effect)  
**M/N** - Increase/Decrease time step by 1 minute (max is 1440 - 1 day)  
**Shift + M/N** - Increase/Decrease time step by 60 minutes (max is 1440 - 1 day)  
**Space** - Toggle the compass globe showing how far the view is turned from default  
**Arrows** - Move view  
**Ctrl + Left/Right** - Rotate view around Moon's polar axis  
**Ctrl + Up/Down** - Rotate view around Moon's equatorial axis  
**Hold and drag left mouse button** - Rotate the eye around Moon  
**Hold and drag right mouse button** - Rotate Moon around the eye (move view)  
**Hold Shift + right mouse button and drag up/down** - Move eye backward/forward  
**Hold Ctrl + drag left mouse button** - Measure distance and elevation difference on Moon surface  
**Hold Shift + left mouse button and drag up/down** - Zoom out/in (more reliable)  
**Mouse wheel up/down** - Zoom in/out (less reliable)  

## Videos
**Sunrise over Montes Apenninus (using time stepping i.e. holding down W key)**  
[![Sunrise](https://img.youtube.com/vi/wd3J2ZEsAoI/maxresdefault.jpg)](https://www.youtube.com/watch?v=wd3J2ZEsAoI)  

**and sunset, exported to MP4 file using F11 key**  
[![Sunset](https://img.youtube.com/vi/C44osfLSB9Y/maxresdefault.jpg)](https://www.youtube.com/watch?v=C44osfLSB9Y)  
This export (using NVENC encoder available on PlotOptiX level) gives high quality (noise-free, no stuttering) output but as video is not captured but generated by renderer, it takes considerable amount of time to export.  

**Libration and apparent size**  
[![Libration](https://img.youtube.com/vi/pmU6eoUkY5o/maxresdefault.jpg)](https://www.youtube.com/watch?v=pmU6eoUkY5o) 

## Screens
**Selenographic grid and labels**
![Grid](images/grid2.jpg)
**Timocharis**
![Timocharis](images/timocharis.jpg)
**Rima Hesiodus and the distant Rupes Recta**
![Rima Hesiodus](images/rima_hesiodus.jpg)
**Maria with [Blue Steel color map](https://astrogeology.usgs.gov/search/map/moon_lro_lola_color_shaded_relief_blue_steel_474m)**
![Maria](images/maria.jpg)
**Mare Smythii in 2 projections (standard and Blue Steel)**
![Mare Smythii](images/mare_smythii.jpg)
**Measuring the depth (Δh) of Theophilus**
![Theophilus](images/theophilus.jpg)
**Field of view frame**
![Theophilus](images/fov.jpg)
**Rise and set window**
![Rise and set window](images/rise_and_set.jpg)





