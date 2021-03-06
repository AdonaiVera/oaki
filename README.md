# OAKI πππ
OAKI is your personal trainer that tries to do the training in home more fun and with metrics to find if you are doing a good exercise.Β 
It combines mediaPipe algorithms,Β depth from the OAKD-lite, functions to find angles and distances between points, some multiprocess to run everything in parallel, and an assistant voice to have a fantastic experience in training.

## Installation π¦
Install libraries necessary for the project
```
pip install -r requirements.txt
```

## Run π’
Run this command to start the app. 
```bash
python app.py
```

## Video π
[![ScreenShot](img/oaki.png?raw=true)](https://www.linkedin.com/feed/update/urn:li:activity:6893223346083811328/)

### Project Layout π₯

As our application grows we would refactor our app.py file into multiple folders and files.

```bash
.
βββ app.py
βββ oakiAgent.py
βββ .gitignore
βββ requirements.txt
βββ README.md
βββ voice
|   βββ 0.mp4
|   βββ 1.mp4
|   βββ 2.mp4
|   βββ 3.mp4
|   βββ 4.mp4
|   βββ 5.mp4
|   βββ 6.mp4
|   βββ 7.mp4
|   βββ 8.mp4
|   βββ 9.mp4
|   βββ 10.mp4
|   βββ 11.mp4
|   βββ 12.mp4
βββ methods
|   βββ body_part_angle.py
|   βββ types_of_exercise.py
|   βββ utils.py
βββ img
```


## Built with  π οΈ
_Mention the tools you used to create your project_
### HARDWARE
* [OpenCV](https://opencv.org/) - OpenCV provides a real-time optimized Computer Vision library, tools, and hardware
* [MediaPipe](https://google.github.io/mediapipe/) - MediaPipe offers cross-platform, customizable ML solutions for live and streaming media.
* [DepthAI](https://docs.luxonis.com/en/latest/) - DepthAI is the embedded spatial AI platform built around Myriad X 

### SOFTWARE
* [NVIDIA GTX960M](https://www.nvidia.com/en-us/geforce/gaming-laptops/geforce-gtx-960m/specifications/) - Run in RealTime
* [OAKD-Lite](https://docs.luxonis.com/projects/hardware/en/latest/pages/DM9095.html) - Depth Camera


## Versions π
We use [SemVer](http://semver.org/) for versioning. For all available versions, see what[tags in this repository](https://github.com/tu/proyecto/tags).

## Authors βοΈ
* **Adonai Vera** - *Member DS4a* - [AdonaiVera](https://github.com/AdonaiVera)

You can also look at the list of all [contributors](https://github.com/AdonaiVera/oaki/contributors) who have participated in this project. 

## License π

This project is under the License, see the file [LICENSE.md](LICENSE.md) more details.

## Expressions of Gratitudeπ

* Tell others about this project π’
* Give thanks publicly π€.

---
