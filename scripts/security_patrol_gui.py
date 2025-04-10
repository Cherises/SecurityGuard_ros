#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import rospy
import os
import subprocess
import threading
import cv2
import tkFileDialog
import tkMessageBox
from datetime import datetime
from Tkinter import *
from ttk import Treeview, Combobox
from geometry_msgs.msg import PointStamped, PoseStamped, Twist, Pose, Point, Quaternion
from darknet_ros_msgs.msg import BoundingBoxes
from tf import TransformListener
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import math
from tf.transformations import quaternion_from_euler
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import *
from tf.transformations import euler_from_quaternion
import pickle

class SecurityPatrolGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("安巡卫士导航系统")
        self.points = []
        rospy.init_node("security_patrol_gui", anonymous=True)
        self.tf_listener = TransformListener()
        self.current_bboxes = []
        self.running = False
        self.bridge = CvBridge()
        self.image_frame = None  # 当前帧图像
        self.move_base_client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        self.move_base_client.wait_for_server()
        self.nav_success = False  # 导航状态标志
        self.current_goal = None  # 记录当前目标坐标
        self.goal_finish = 0  # 目标完成标志
        self.cancel_requested = False  # 标记是否主动取消

        self.person_detected = False  # 行人检测标志
        self.person_detection_thread = None  # 行人检测线程

        self.setup_gui()
        
        rospy.Subscriber("/clicked_point", PointStamped, self.point_callback)
        rospy.Subscriber("/darknet_ros/bounding_boxes", BoundingBoxes, self.bbox_callback)
        rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback)
        self.goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=10)
        self.stop_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

    def setup_gui(self):
        self.left_frame = Frame(self.root)
        self.left_frame.pack(side=LEFT, fill=Y)

        self.tree = Treeview(self.left_frame, columns=('X', 'Y', 'Class'), show='headings')
        self.tree.heading('X', text='X 坐标')
        self.tree.heading('Y', text='Y 坐标')
        self.tree.heading('Class', text='目标类别')
        self.tree.pack(fill=Y)

        self.save_btn = Button(self.left_frame, text="保存列表", command=self.save_points)
        self.save_btn.pack(pady=2)

        self.load_btn = Button(self.left_frame, text="加载列表", command=self.load_points)
        self.load_btn.pack(pady=2)

        Label(self.left_frame, text="X坐标:").pack()
        self.x_entry = Entry(self.left_frame)
        self.x_entry.pack()

        Label(self.left_frame, text="Y坐标:").pack()
        self.y_entry = Entry(self.left_frame)
        self.y_entry.pack()

        Label(self.left_frame, text="目标名:").pack()
        self.class_entry = Entry(self.left_frame)
        self.class_entry.pack()
        
        self.add_manual_btn = Button(self.left_frame, text="添加手动点", command=self.add_manual_point)
        self.add_manual_btn.pack(pady=2)

        self.right_frame = Frame(self.root)
        self.right_frame.pack(side=RIGHT, fill=BOTH, expand=True)

        self.start_btn = Button(self.right_frame, text="开始巡逻", command=self.start_patrol_thread)
        self.start_btn.pack(pady=5)

        self.stop_btn = Button(self.right_frame, text="急停", command=self.stop_robot)
        self.stop_btn.pack(pady=5)

        self.open_folder_btn = Button(self.right_frame, text="打开照片文件夹", command=self.open_folder)
        self.open_folder_btn.pack(pady=5)

        self.log_text = Text(self.right_frame, height=20)
        self.log_text.pack(fill=BOTH, expand=True)

        self.clear_btn = Button(self.right_frame, text="清空坐标列表", command=self.clear_points)
        self.clear_btn.pack(pady=5)

    def save_points(self):
        filename = tkFileDialog.asksaveasfilename(defaultextension=".sg", filetypes=[("Saved Points", "*.sg")])
        if filename:
            try:
                with open(filename, "wb") as f:
                    pickle.dump(self.points, f)
                self.log("坐标列表已保存到: {}".format(filename))
            except Exception as e:
                self.log("保存失败: {}".format(e))

    def load_points(self):
        filename = tkFileDialog.askopenfilename(defaultextension=".sg", filetypes=[("Saved Points", "*.sg")])
        if filename:
            try:
                with open(filename, "rb") as f:
                    self.points = pickle.load(f)
                for item in self.tree.get_children():
                    self.tree.delete(item)
                for x, y, cls in self.points:
                    self.tree.insert('', 'end', values=(round(x, 2), round(y, 2), cls))
                self.log("坐标列表已加载自: {}".format(filename))
            except Exception as e:
                self.log("加载失败: {}".format(e))

    def add_manual_point(self):
        try:
            x = float(self.x_entry.get())
            y = float(self.y_entry.get())
            cls = self.class_entry.get()
            if not cls:
                tkMessageBox.showwarning("输入错误", "请输入目标类别")
                return
            self.points.append((x, y, cls))
            self.tree.insert('', 'end', values=(round(x, 2), round(y, 2), cls))
            self.log("手动添加坐标点：({}, {}, {})".format(x, y, cls))
        except ValueError:
            tkMessageBox.showerror("输入错误", "请输入有效的数字坐标")

    def image_callback(self, msg):
        try:
            self.image_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.log("图像转换失败: {}".format(e))

    def clear_points(self):
        self.points = []  # 清空内部数据列表
        for item in self.tree.get_children():
            self.tree.delete(item)  # 清空 Treeview 中的所有项
        self.log("已清空所有坐标点。")

    def point_callback(self, msg):
        new_window = Toplevel(self.root)
        new_window.title("选择物体类别")

        Label(new_window, text="为坐标({:.2f}, {:.2f})选择类别：".format(msg.point.x, msg.point.y)).pack()
        class_var = StringVar(new_window)
        class_combobox = Combobox(new_window, textvariable=class_var)
        class_combobox['values'] = ("chair", "bottle", "tvmonitor", "cup", "person")
        class_combobox.current(0)
        class_combobox.pack()

        def confirm():
            selected_class = class_combobox.get()
            self.points.append((msg.point.x, msg.point.y, selected_class))
            self.tree.insert('', 'end', values=(round(msg.point.x, 2), round(msg.point.y, 2), selected_class))
            self.log("添加坐标点：({}, {})，类别：{}".format(msg.point.x, msg.point.y, selected_class))
            new_window.destroy()

        Button(new_window, text="确认", command=confirm).pack()

    def bbox_callback(self, msg):
        self.current_bboxes = msg.bounding_boxes

    def start_patrol_thread(self):
        if not self.running:
            threading.Thread(target=self.start_patrol).start()

    def start_patrol(self):
        if not self.points:
            self.log("无巡逻目标，请先添加坐标点。")
            return
        self.running = True
        self.cancel_requested = False  # 重置取消标志
        self.person_detected = False

        # 启动行人检测线程
        self.person_detection_thread = threading.Thread(target=self.monitor_person_detection)
        self.person_detection_thread.start()

        for idx, (target_x, target_y, target_class) in enumerate(self.points):
            if not self.running or self.cancel_requested:
                break

            # 创建目标点(直接使用目标坐标)
            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = 'map'
            goal.target_pose.header.stamp = rospy.Time.now()
            goal.target_pose.pose.position.x = target_x
            goal.target_pose.pose.position.y = target_y
            goal.target_pose.pose.orientation.w = 1.0  # 默认朝向

            self.log("开始导航到目标 ({:.2f}, {:.2f})".format(target_x, target_y))
            self.speak("start")

            # 发送目标但不等待完成
            self.move_base_client.send_goal(goal)
            
            # 实时监测位置
            rate = rospy.Rate(10)  # 10Hz
            reached = False
            while not rospy.is_shutdown() and self.running and not reached:
                try:
                    # 获取当前位置
                    (current_x, current_y, _) = self.get_current_pose()
                    
                    # 计算到目标的距离
                    distance = math.sqrt((target_x-current_x)**2 + (target_y-current_y)**2)
                    
                    # 当进入1米范围内时停止
                    if distance < 0.7:
                        self.move_base_client.cancel_goal()
                        self.stop_robot(soft_stop=True)
                        reached = True
                        self.log("已进入目标1米范围，停止导航")
                        
                except Exception as e:
                    self.log("位置监测错误: {}".format(e))
                    break
                    
                rate.sleep()

            if reached:
                # 检查目标物体
                found = self.check_object(target_class)
                if found:
                    self.log("检测到目标物体：{}".format(target_class))
                    self.speak("success")
                else:
                    self.log("未检测到目标物体：{}".format(target_class))
                    self.speak("failed")

            rospy.sleep(1.0)


        self.running = False
        if self.person_detection_thread is not None:
            self.person_detection_thread.join()
        self.log("巡逻完成。")
        self.speak("end")

    def get_current_pose(self):
        """获取当前机器人在map坐标系中的位置"""
        try:
            now = rospy.Time.now()
            self.tf_listener.waitForTransform("map", "base_link", now, rospy.Duration(4.0))
            (trans, rot) = self.tf_listener.lookupTransform("map", "base_link", now)
            return (trans[0], trans[1], euler_from_quaternion(rot)[2])  # x, y, yaw
        except Exception as e:
            raise Exception("获取当前位姿失败: {}".format(e))

    def donecb(self, status, result):
        """完成回调"""
        if status == GoalStatus.SUCCEEDED:
            self.log("导航成功到达目标点")
            self.goal_finish = 1
        elif status == GoalStatus.PREEMPTED and not self.cancel_requested:
            self.log("导航被意外取消")
            self.goal_finish = 0
        else:
            self.log("导航未能到达目标点，状态码: {}".format(status))
            self.goal_finish = 0

    def activecb(self):
        """活动回调"""
        self.log("开始向目标点移动...")
        self.cancel_requested = False  # 重置取消标志

    def feedbackcb(self, feedback):
        """反馈回调"""
        # 可以添加反馈处理逻辑
        pass

    def wait_for_done(self):
        """等待完成函数"""
        rate = rospy.Rate(10) 
        start_time = rospy.Time.now()
        while not rospy.is_shutdown() and self.running and not self.cancel_requested:
            if self.goal_finish == 1:
                return True
            if (rospy.Time.now() - start_time).to_sec() > 60:  # 60秒超时
                self.log("导航超时")
                return False
            rate.sleep()
        return False

    def stop_robot(self, soft_stop=False):
        """停止机器人
        soft_stop: True-仅停止移动不取消目标，False-取消目标并停止
        """
        if not soft_stop:
            self.cancel_requested = True
            self.move_base_client.cancel_all_goals()
            self.log("取消所有目标")
        twist = Twist()  # 发送空的速度指令
        self.stop_pub.publish(twist)
        self.log("机器人已停止" + (" (保留目标)" if soft_stop else " (取消目标)"))

    def check_object(self, target_class, timeout=5.0):
        """等待 timeout 秒内持续检测目标"""
        start_time = rospy.Time.now().to_sec()
        while rospy.Time.now().to_sec() - start_time < timeout:
            for box in self.current_bboxes:
                if box.Class == target_class and box.probability > 0.25:
                    return True
            rospy.sleep(0.2)
        return False
    
    def monitor_person_detection(self):
        """持续检测行人的线程函数"""
        while self.running and not rospy.is_shutdown():
            for box in self.current_bboxes:
                if box.Class == "person" and box.probability > 0.9 and not self.person_detected:
                    self.log("检测到行人，正在保存图像...")
                    self.speak("person")
                    self.save_person_image()
                    self.person_detected = True
                    # 重置检测标志，以便检测到新人时可以再次保存
                    rospy.sleep(2)  # 避免连续检测同一人
                    self.person_detected = False
            rospy.sleep(0.3)

    def save_person_image(self):
        if self.image_frame is not None:
            folder = "/home/handsfree/catkin_ws/src/security_guard/src/person/"
            if not os.path.exists(folder):
                os.makedirs(folder)
            filename = datetime.now().strftime("%Y%m%d_%H%M%S.jpg")
            path = os.path.join(folder, filename)
            cv2.imwrite(path, self.image_frame)
            self.log("已保存照片：{}".format(path))
        else:
            self.log("图像帧为空，无法保存。")

    def open_folder(self):
        folder = "/home/handsfree/catkin_ws/src/security_guard/src/person/"
        if not os.path.exists(folder):
            os.makedirs(folder)
        subprocess.Popen(['xdg-open', folder])

    def speak(self, message):
        #message: start  end  success  failed  person
        voice_dir = "/home/handsfree/catkin_ws/src/security_guard/src/voice/"
        mp3_path = os.path.join(voice_dir, message + ".mp3")
        os.system("play " + mp3_path)

    def log(self, message):
        self.log_text.insert(END, message + "\n")
        self.log_text.see(END)
    

if __name__ == "__main__":
    root = Tk()
    app = SecurityPatrolGUI(root)
    root.mainloop()