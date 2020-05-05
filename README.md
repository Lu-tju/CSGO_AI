# CSGO_AI

基于YOLOv3的csgo自瞄外挂
----------------------

CSGO.py是自己改写的。其余注释及代码来自的：https://blog.csdn.net/qq_34199326/article/details/84072505 ，并按照自己的理解对注释稍加修改

原文件：https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch

直接运行CSGO.py即可

csgo设置：

·鼠标设置中：灵敏度调8，关闭原始数据输入

·视频设置中：窗口化运行，并拖放到屏幕左上角（我的移动鼠标的参数是根据这里的坐标写的，不然移动鼠标不准，大家可以自己测试下全屏时，移动鼠标的输入   量和准星的移动对应关系）

python库：

·获取截图：PIL

·目标检测：YOLOv3相关依赖：numpy opencv-python pytorch等，参见原文

·控制鼠标：pyautogui


效果参见：https://www.bilibili.com/video/BV1v5411x7Mr?p=1
