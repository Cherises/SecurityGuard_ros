# 安巡卫士机器人系统

## 一、系统概述  
安巡卫士机器人系统是一个基于ROS（Robot Operating System）平台开发的智能巡逻机器人平台，集成了图像识别、多目标导航、人员检测、地图标记、语音播报和GUI操作等功能，具备较强的实用性和扩展性。  

整个系统以Tkinter为前端控制界面，通过调用ROS节点来完成各类巡逻任务，能够自主规划路径、识别场景目标并进行动态记录和播报，满足仓库、教室等场所的日常智能巡逻需求。

## 二、系统架构  
系统主要包括以下几个模块：

### GUI控制模块  
- **技术栈**: 基于Tkinter  
- **功能**: 实现用户与机器人之间的交互，控制巡逻任务的开始、停止、目标点设置等功能  

### 导航模块  
- **技术栈**: 基于move_base  
- **功能**: 实现路径规划和自主导航  

### 目标识别模块  
- **技术栈**: 基于darknet_ros  
- **功能**: 使用YOLO算法对摄像头采集的图像进行实时分析，识别包括人物、物品等目标  

### 地图标记与路径记录模块  
- **功能**:  
  - 在RViz中使用Marker功能对识别目标进行标注  
  - 记录目标位置与类别  
  - 辅助后续导航任务  

### 语音播报模块  
- **技术**: 使用pyttsx语音引擎  
- **功能**: 对识别目标进行语音播报  

### TF变换与多传感器融合模块  
- **功能**:  
  - 利用TF广播机制将目标识别位置从相机坐标转换至map坐标系  
  - 融合深度图与IMU信息提高定位精度  

## 系统特性  
✅ 多目标自主导航  
✅ 实时物体识别  
✅ 可视化地图标记  
✅ 智能语音交互  
✅ 高精度定位  
