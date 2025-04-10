#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import rospy
from darknet_ros_msgs.msg import BoundingBoxes
import Tkinter as tk
import ttk
from datetime import datetime

class DetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("物体检测信息展示")
        self.root.geometry("600x400")

        rospy.init_node('detection_gui_node', anonymous=True)

        self.detected_items = set()  # 存储已检测物品，防止重复
        self.last_detected_time = rospy.Time.now()  # 记录最近一次检测时间
        self.throttle_time = rospy.Duration(1.0)  # 设定节流时间为 1 秒

        self.create_widgets()

        self.sub_boxes = rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, self.bounding_box_callback)

    def create_widgets(self):
        self.tree = ttk.Treeview(self.root, columns=('ID', '名称', '置信度', 'X', 'Y'), show='headings')
        for col in ('ID', '名称', '置信度', 'X', 'Y'):
            self.tree.heading(col, text=col)
        self.tree.pack(fill=tk.BOTH, expand=True, pady=5)

        input_frame = ttk.Frame(self.root)
        input_frame.pack(pady=5)

        ttk.Label(input_frame, text="物品ID:").grid(row=0, column=0)
        self.id_entry = ttk.Entry(input_frame)
        self.id_entry.grid(row=0, column=1)

        ttk.Label(input_frame, text="名称:").grid(row=1, column=0)
        self.name_entry = ttk.Entry(input_frame)
        self.name_entry.grid(row=1, column=1)

        ttk.Label(input_frame, text="X:").grid(row=2, column=0)
        self.x_entry = ttk.Entry(input_frame)
        self.x_entry.grid(row=2, column=1)

        ttk.Label(input_frame, text="Y:").grid(row=3, column=0)
        self.y_entry = ttk.Entry(input_frame)
        self.y_entry.grid(row=3, column=1)

        self.save_button = ttk.Button(input_frame, text="保存坐标", command=self.save_coordinates)
        self.save_button.grid(row=4, column=0, columnspan=2, pady=5)

        self.clear_button = ttk.Button(input_frame, text="清空列表", command=self.clear_list)
        self.clear_button.grid(row=5, column=0, columnspan=2, pady=5)

    def bounding_box_callback(self, msg):
        current_time = rospy.Time.now()
        if current_time - self.last_detected_time < self.throttle_time:
            return  # 如果距离上次检测时间小于1秒，则忽略本次数据

        self.last_detected_time = current_time  # 更新检测时间

        for box in msg.bounding_boxes:
            if box.probability >= 0.90:
                if box.Class.lower() == "person":
                    self.capture_image()
                else:
                    # 避免重复检测相同物体
                    item_key = (box.Class, round(box.probability, 2))
                    if item_key in self.detected_items:
                        continue  # 已经检测过该物品，跳过

                    self.detected_items.add(item_key)  # 添加到已检测物品集合
                    self.tree.insert(
                        '', 'end',
                        values=(
                            len(self.detected_items),
                            box.Class,
                            "{:.1f}%".format(box.probability * 100),
                            "待输入",
                            "待输入"
                        )
                    )

    def save_coordinates(self):
        item_id = self.id_entry.get()
        name = self.name_entry.get()
        x = self.x_entry.get()
        y = self.y_entry.get()

        if item_id and name and x and y:
            for item in self.tree.get_children():
                values = self.tree.item(item, 'values')
                if values[0] == item_id:
                    self.tree.item(item, values=(item_id, name, values[2], x, y))
                    break

    def clear_list(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.detected_items.clear()  # 清空已检测物品记录

    def capture_image(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = "person_detected_{}.jpg".format(timestamp)
        rospy.loginfo("检测到人员，拍照并保存至 {}".format(filename))

    def on_closing(self):
        self.sub_boxes.unregister()
        self.root.destroy()

if __name__ == '__main__':
    try:
        root = tk.Tk()
        app = DetectionGUI(root)
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        root.mainloop()
    except rospy.ROSInterruptException:
        pass

