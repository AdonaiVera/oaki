# OAKI 🚀🚀🚀
OAKI is your personal trainer that tries to do the training in home more fun and with metrics to find if you are doing a good exercise. 
It combines mediaPipe algorithms, depth from the OAKD-lite, functions to find angles and distances between points, some multiprocess to run everything in parallel, and an assistant voice to have a fantastic experience in training.

## Installation 📦
Install libraries necessary for the project
```
pip install -r requirements.txt
```

## Run 📢
Run this command to start the app. 
```bash
python app.py
```

## Video 📖
[![ScreenShot](img/oaki.png?raw=true)](https://www.linkedin.com/feed/update/urn:li:activity:6893223346083811328/)

### Project Layout 🖥

As our application grows we would refactor our app.py file into multiple folders and files.

```bash
.
├── app.py
├── oakiAgent.py
├── .gitignore
├── requirements.txt
├── README.md
├── voice
|   ├── 0.mp4
|   ├── 1.mp4
|   ├── 2.mp4
|   ├── 3.mp4
|   ├── 4.mp4
|   ├── 5.mp4
|   ├── 6.mp4
|   ├── 7.mp4
|   ├── 8.mp4
|   ├── 9.mp4
|   ├── 10.mp4
|   ├── 11.mp4
|   └── 12.mp4
├── methods
|   ├── body_part_angle.py
|   ├── types_of_exercise.py
|   └── utils.py
└── img
```


## Built with  🛠️
_Mention the tools you used to create your project_
### HARDWARE
* [OpenCV](https://opencv.org/) - OpenCV provides a real-time optimized Computer Vision library, tools, and hardware
* [MediaPipe](https://google.github.io/mediapipe/) - MediaPipe offers cross-platform, customizable ML solutions for live and streaming media.
* [DepthAI](https://docs.luxonis.com/en/latest/) - DepthAI is the embedded spatial AI platform built around Myriad X 

### SOFTWARE
* [NVIDIA GTX960M](https://www.nvidia.com/en-us/geforce/gaming-laptops/geforce-gtx-960m/specifications/) - Run in RealTime
* [OAKD-Lite](https://docs.luxonis.com/projects/hardware/en/latest/pages/DM9095.html) - Depth Camera


## Versions 📌
We use [SemVer](http://semver.org/) for versioning. For all available versions, see what[tags in this repository](https://github.com/tu/proyecto/tags).

## Authors ✒️
* **Adonai Vera** - *Member DS4a* - [AdonaiVera](https://github.com/AdonaiVera)

You can also look at the list of all [contributors](https://github.com/AdonaiVera/oaki/contributors) who have participated in this project. 

## License 📄

This project is under the License, see the file [LICENSE.md](LICENSE.md) more details.

## Expressions of Gratitude🎁

* Tell others about this project 📢
* Give thanks publicly 🤓.

---
