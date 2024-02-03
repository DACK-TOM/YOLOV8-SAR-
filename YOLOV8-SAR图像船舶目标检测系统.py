from PyQt6.QtWidgets import  QMainWindow
from PyQt6 import QtCore, QtGui, QtWidgets
import os
import shutil
import cv2
from PIL import Image
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QPixmapCache, QBrush
from PyQt6.QtWidgets import QApplication, QLineEdit, QPushButton, QRadioButton, QFileDialog, QLabel, QTabWidget, \
    QWidget,  QMessageBox, QProgressBar
import sys

from torch import multiprocessing
from ultralytics import YOLO
def pre(in_model_path, in_images_path, out_path):
    progressBar.reset()
    current_index = tabWidget.currentIndex()
    print(current_index)
    global stop
    global iou, conf, save_txt, save_conf, save_crop, exist_ok
    global pixmap0, pixmap1, pixmap2, pixmap3, pixmap4
    stop = False
    if in_model_path == '':
        lineEdit.setText(lineEdit.placeholderText())
        in_model_path = lineEdit.placeholderText()
    if in_images_path == '':
        lineEdit_2.setText(lineEdit_2.placeholderText())
        in_images_path = lineEdit_2.placeholderText()
    if out_path == '':
        lineEdit_3.setText(lineEdit_3.placeholderText())
        out_path = lineEdit_3.placeholderText()
    if current_index == 0:
        # 图片被保存在这个路径
        output_name = in_images_path.split('/')
        output_name = output_name[-1]  # 不带路径的纯文件名字，例 000006.jpg
        suffix=['bmp' ,'dng' ,'jpeg' ,'jpg', 'mpo' ,'png', 'tif', 'tiff' ,'webp', 'pfm']
        suffix_ok=False
        if '.'in output_name:
            output_name_suffix=output_name.split('.')[1]
            print('output_name_suffix',output_name_suffix)
            for suffix_one in suffix:
                if output_name_suffix ==suffix_one:
                    suffix_ok=True
        if suffix_ok ==False:
            error_message()
            return 0
        print('output_name', output_name)
        model = YOLO(in_model_path)
        if save_txt or save_crop:
            name = output_name.split('.')[0]
            name = f'{out_path}/{name}'  # 如果额外保存txt文件就不能直接在输出文件夹内放入图片，要新建个文件夹放图片及txt
            results = model(in_images_path, iou=iou, save_txt=save_txt,
                            save_conf=save_conf, name=name, exist_ok=exist_ok, save_crop=save_crop, conf=conf,
                            show=False)
        else:
            results = model(in_images_path, iou=iou, exist_ok=exist_ok, conf=conf, show=False)
        for r in results:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            if save_txt or save_crop:
                name = output_name.split('.')[0]
                name = f'{out_path}/{name}'
                output = f'{name}/{output_name}'
            else:
                output = f'{out_path}/{output_name}'
            im.save(output)  # save image with new name
            progressBar_work(100) # 更新进度条
            if show_value:
                # print('之前',pixmap3)
                pixmap3 = QPixmap(in_images_path)
                print('之后', pixmap3)
                pixmap3 = pixmap3.scaled(label_8.width(), label_8.height())
                label_8.setPixmap(pixmap3)
                # 创建一个QPixmap对象并加载图片
                pixmap4 = QPixmap(output)
                pixmap4 = pixmap4.scaled(label_7.width(), label_7.height())
                # 将图片设置到label_3中
                label_7.setPixmap(pixmap4)
    if current_index == 1:  # 第二页对应读取视频模式
        output_name = in_images_path.split('/')
        output_name = output_name[-1]
        suffix = ['asf' ,'avi' ,'gif','m4v' ,'mkv' ,'mov' ,'mp4','mpeg' ,'mpg' ,'ts' ,'wmv' ,'webm']
        suffix_ok = False
        if '.' in output_name:
            output_name_suffix = output_name.split('.')[1]
            print('output_name_suffix', output_name_suffix)
            for suffix_one in suffix:
                if output_name_suffix == suffix_one:
                    suffix_ok = True
        if suffix_ok == False:
            error_message()
            return 0
        cap = cv2.VideoCapture(in_images_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print('total_frames',total_frames)
        model = YOLO(in_model_path)
        i = 0
        output_name = output_name.split('.')[0]
        output_name = f'{output_name}_video'  # 防止输入视频名为全数字和文件整理函数冲突造成误删崩溃
        output_dir = f'{out_path}/{output_name}'
        print('输出视频结果地址', output_dir)
        if save_txt or save_crop:
            output_img_dir = f'{output_dir}/{output_name}'  # 视频帧临时保存文件夹
        else:
            output_img_dir = output_dir
        os.makedirs(output_img_dir, exist_ok=True)
        while cap.isOpened():
            i_long = str(i).zfill(8)
            name = f'{output_dir}/{i_long}'  # 如果额外保存txt文件就不能直接在输出文件夹内放入图片，要新建个文件夹放图片及txt
            output = f'{output_img_dir}/{i_long}.jpg'  # 视频帧
            ret, frame = cap.read()
            if not ret:
                break
            if save_txt or save_crop:
                print('name', name)
                results = model(frame, iou=iou, save_txt=save_txt,
                                save_conf=save_conf, name=name, save_crop=save_crop, conf=conf, exist_ok=exist_ok,
                                show=False)
                print('保存txt路径', name)
            else:
                results = model(frame, iou=iou, conf=conf, exist_ok=exist_ok, show=False)

            for r in results:
                im_array = r.plot()  # plot a BGR numpy array of predictions
                im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
                im.save(output)  # save image with new name
                if show_value:
                    QPixmapCache.clear()
                    pixmap2 = QPixmap(output)
                    pixmap2 = pixmap2.scaled(label_6.width(), label_6.height())
                    # 将图片设置到label_6中
                    label_6.setPixmap(pixmap2)
                i = i + 1
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
            if stop:
                break
            progressBar_work(int(i * 100 / total_frames))
        cap.release()
        cv2.destroyAllWindows()
        save_video(output_img_dir)
        file_ordering(output_dir, save_txt=save_txt, save_crop=save_crop)
    if current_index == 2:  # 第三页对应读取文件夹模式，文件夹只能有图片
        output_name = in_images_path.split('/')
        output_name = output_name[-1]   # 纯输出文件夹名
        suffix_ok = True
        if '.' in output_name:
            suffix_ok = False
        if suffix_ok == False:
            error_message()
            return 0
        filenames = os.listdir(in_images_path)
        print(f'"输入文件夹："{in_images_path}')
        output_dir = f'{out_path}/{output_name}'  # 输出文件夹路径
        os.makedirs(output_dir, exist_ok=True)
        print('output_dir', output_dir)
        model = YOLO(in_model_path)
        i=0
        for one_image in filenames:
            if stop:
                break
            file_count = len(filenames)
            in_img = f'{in_images_path}/{one_image}'  # 单个图片输入地址
            if save_txt or save_crop:
                one_image_name = one_image.split('.')[0]
                name = f'{output_dir}/{one_image_name}'
                results = model(in_img, iou=iou, save_txt=save_txt,
                                save_conf=save_conf, name=name, save_crop=save_crop, conf=conf, exist_ok=exist_ok,
                                show=False)
                print('保存txt路径', name)
            else:
                results = model(in_img, iou=iou, conf=conf, exist_ok=exist_ok, show=False)
            for r in results:
                im_array = r.plot()  # plot a BGR numpy array of predictions
                im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
                # 图片被保存在这个路径
                output = f'{output_dir}/{one_image}'
                im.save(output)  # save image with new name
                print(f'"输出",{output}')
                if show_value:
                    QPixmapCache.clear()
                    pixmap0 = QPixmap(in_img)
                    pixmap0 = pixmap0.scaled(label_3.width(), label_3.height())
                    label_3.setPixmap(pixmap0)
                    if cv2.waitKey(1) & 0xff == ord('q'):
                        break
                    # 创建一个QPixmap对象并加载图片
                    QPixmapCache.clear()
                    pixmap1 = QPixmap(output)
                    pixmap1 = pixmap1.scaled(label_5.width(), label_5.height())
                    # 将图片设置到label_3中
                    label_5.setPixmap(pixmap1)
                    if cv2.waitKey(1) & 0xff == ord('q'):
                        break
            i=i+1
            progressBar_work(int(100 * i / file_count))
        file_ordering(output_dir, save_txt=save_txt, save_crop=save_crop)
    if current_index == 3:
        error_message()
def file_ordering(path, save_txt=False, save_crop=False):
    # 记录需要删除的文件夹路径
    folders_to_delete = set()
    # 判断是否需要处理 labels 文件夹
    if save_txt:
        for root, dirs, files in os.walk(path):
            for directory in dirs:
                if directory.isdigit():  # 检查文件夹名称是否为数字
                    txt_folder_path = os.path.join(root, directory, 'labels')
                    if os.path.exists(txt_folder_path):
                        # 新建 labels 文件夹并移动 .txt 文件
                        new_labels_folder_path = os.path.join(root, 'labels')
                        os.makedirs(new_labels_folder_path, exist_ok=True)
                        for txt_file in os.listdir(txt_folder_path):
                            if txt_file.endswith('.txt'):
                                old_file_path = os.path.join(txt_folder_path, txt_file)
                                new_file_path = os.path.join(new_labels_folder_path, f'{directory}.txt')
                                shutil.move(old_file_path, new_file_path)
                                print(f'Moved: {old_file_path} -> {new_file_path}')
                        # 删除原始的 labels 文件夹
                        shutil.rmtree(txt_folder_path)
                        print(f'Deleted folder: {txt_folder_path}')
                        # 记录需要删除的文件夹路径
                        folders_to_delete.add(os.path.join(root, directory))
    # 判断是否需要处理 crops 文件夹
    if save_crop:
        for root, dirs, files in os.walk(path):
            for directory in dirs:
                if directory.isdigit():  # 检查文件夹名称是否为数字命名的文件夹
                    source_image_folder = os.path.join(root, directory, 'crops', 'ship')
                    if os.path.exists(source_image_folder):
                        # 获取新的文件名
                        new_file_name = f'{directory}_ship.jpg'
                        # 设置目标文件夹路径
                        destination_image_folder = os.path.join(root, 'crops')
                        os.makedirs(destination_image_folder, exist_ok=True)
                        # 移动并重命名图片
                        source_image_path = os.path.join(source_image_folder, os.listdir(source_image_folder)[0])
                        destination_image_path = os.path.join(destination_image_folder, new_file_name)
                        shutil.move(source_image_path, destination_image_path)
                        print(f'Moved and renamed: {source_image_path} -> {destination_image_path}')
                        # 记录需要删除的文件夹路径
                        folders_to_delete.add(os.path.join(root, directory))
    # 删除所有记录的文件夹
    for folder in folders_to_delete:
        shutil.rmtree(folder)
        print(f'Deleted folder: {folder}')
def save_video(out_video_path):
    # 图片文件夹路径
    image_folder = out_video_path
    # 视频文件保存路径和名称
    video_name = f'{out_video_path}.mp4'
    print('视频保存路径', video_name)
    # 获取图片文件夹中的所有图片文件
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")]
    # 获取图片的宽度和高度（假设所有图片的大小相同）
    image_path = os.path.join(image_folder, images[0])
    img = cv2.imread(image_path)
    height, width, _ = img.shape
    # 设置视频编码器和帧速率
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或者使用 cv2.VideoWriter_fourcc(*'H264')
    video = cv2.VideoWriter(video_name, fourcc, 24, (width, height))
    # 将每张图片逐一写入视频
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        # 将图像写入视频文件
        video.write(frame)
    # 关闭视频文件
    video.release()
    # 删除图片文件夹
    shutil.rmtree(out_video_path)
    print("视频已成功生成：", video_name)
def show():
    global show_value
    a = 0
    if (show_value == False) and (a == 0):
        show_value = True
        a = 1
    if (show_value == True) and (a == 0):
        show_value = False
    print('show_value', show_value)
def open_folder_dialog(n):
    current_index = tabWidget.currentIndex()
    global select_in_dir_path
    if n == 3:
        if current_index == 0:
            fd = QFileDialog()
            fd.setFileMode(QFileDialog.FileMode.AnyFile)  # 设置只选择文件
            fd.setDirectory('c:\\')  # 设置初始化路径
            fd.setNameFilter('images (*.bmp *.dng *.jpeg *.jpg *.mpo *.png *.tif *.tiff *.webp *.pfm )')  # 设置文件过滤器
            if fd.exec():
                select_in_dir_path = fd.selectedFiles()
                # 这里的files是一个包含了所有选中文件路径的列表
                Renewal(n)
        if current_index == 1:
            fd = QFileDialog()
            fd.setFileMode(QFileDialog.FileMode.AnyFile)  # 设置只选择文件
            fd.setDirectory('c:\\')  # 设置初始化路径
            fd.setNameFilter(
                'Videos (*.asf *.avi *.gif *.m4v *.mkv *.mov *.mp4 *.mpeg *.mpg *.ts *.wmv *.webm)')  # 设置文件过滤器
            if fd.exec():
                select_in_dir_path = fd.selectedFiles()
                # 这里的files是一个包含了所有选中文件路径的列表
                Renewal(n)
        if current_index == 2:
            fd = QFileDialog()
            fd.setFileMode(QFileDialog.FileMode.Directory)  # 设置只选择目录
            fd.setDirectory('c:\\')  # 设置初始化路径
            if fd.exec():
                select_in_dir_path = fd.selectedFiles()
                Renewal(n)
    if n == 4:
        fd = QFileDialog()
        fd.setFileMode(QFileDialog.FileMode.Directory)  # 设置只选择目录
        fd.setDirectory('c:\\')  # 设置初始化路径
        if fd.exec():
            select_in_dir_path = fd.selectedFiles()
            Renewal(n)
    if n == 5:
        fd = QFileDialog()
        fd.setFileMode(QFileDialog.FileMode.AnyFile)  # 设置只选择文件
        fd.setDirectory('c:\\')  # 设置初始化路径
        fd.setNameFilter('模型文件 (*.pt )')  # 设置文件过滤器
        if fd.exec():
            select_in_dir_path = fd.selectedFiles()
            # 这里的files是一个包含了所有选中文件路径的列表
            Renewal(n)
def Renewal(n):
    n = n
    global select_in_dir_path
    if n == 3:
        lineEdit_2.setText(select_in_dir_path[0] if select_in_dir_path else '')
    if n == 4:
        lineEdit_3.setText(select_in_dir_path[0] if select_in_dir_path else '')
    if n == 5:
        lineEdit.setText(select_in_dir_path[0] if select_in_dir_path else '')
def stop_main():
    global stop
    stop = True
def update_configuration(up):
    global iou, conf, save_txt, save_conf, save_crop, exist_ok
    iou_default = 0.7
    conf_default = 0.25
    save_txt_default = False
    save_conf_default = False
    save_crop_default = False
    exist_ok_default = True
    if up == 0:  # 更新参数
        if lineEdit_4.text(): iou = float(lineEdit_4.text())
        if lineEdit_5.text(): conf = float(lineEdit_5.text())
        if lineEdit_6.text(): save_txt = lineEdit_6.text() == "True"
        if lineEdit_7.text(): save_conf = lineEdit_7.text() == "True"
        if lineEdit_8.text(): save_crop = lineEdit_8.text() == "True"
    if up == 1:  # 恢复默认配置
        iou = iou_default
        conf = conf_default
        save_txt = save_txt_default
        save_conf = save_conf_default
        save_crop = save_crop_default
        exist_ok = exist_ok_default
    lineEdit_4.setText(str(iou))
    lineEdit_5.setText(str(conf))
    lineEdit_6.setText(str(save_txt))
    lineEdit_7.setText(str(save_conf))
    lineEdit_8.setText(str(save_crop))
def background():
    background_img = 'D:/桌面1/ship1_27.jpg'
    pixmap = QPixmap(background_img)
    pixmap = pixmap.scaled(MainWindow.width(), MainWindow.height())
    palette = MainWindow.palette()
    palette.setBrush(MainWindow.backgroundRole(), QBrush(pixmap))
    MainWindow.setPalette(palette)  # 设置背景
    # 设置登录界面背景，并显示在所有控件前面
    widget.resize(MainWindow.width(), MainWindow.height())
    label_20.resize(widget.width(), widget.height())
    label_20.setPixmap(pixmap)
    widget.raise_()
def log_in(user, password):
    global password_ok, user_ok
    if user == 'admin':
        user_ok = True
    if password == '123':
        password_ok = True
    if user_ok and password_ok:
        widget.hide()
    else :
        result=QMessageBox.information(None,'错误','账户或密码错误',QMessageBox.StandardButton.Ok)       #错误提示对话框
        if result == QMessageBox.StandardButton.Ok:
            lineEdit_11.setText('')
def change_size():  # 窗口自适应
    background()
    global w, h, old_w, old_h
    global pixmap0, pixmap1, pixmap2, pixmap3, pixmap4  # 图片自适应
    old_w = w
    old_h = h
    w = MainWindow.width()
    h = MainWindow.height()
    if (w != old_w) or (h != old_h):
        print('窗口变化了')
        print('ui.width', w)
        print('ui.height', h)
        background_img = 'D:/桌面1/ship1_27.jpg'
        pixmap = QPixmap(background_img)
        pixmap = pixmap.scaled(MainWindow.width(), MainWindow.height())
        palette = MainWindow.palette()
        palette.setBrush(MainWindow.backgroundRole(), QBrush(pixmap))
        MainWindow.setPalette(palette)  # 设置背景
        #
        # 按比例设置三个标签页形状
        tabWidget_x = int(w / 20)
        tabWidget_y = int(h / 80)
        tabWidget_w = int(18 * w / 20)
        tabWidget_h = int(16 * h / 20)
        tabWidget.setGeometry(tabWidget_x, tabWidget_y, tabWidget_w, tabWidget_h)
        #
        #设置操作框大小
        widget_1_x = int(w / 10)
        widget_1_y = int(tabWidget_h + tabWidget_y)
        widget_1_w = int(8 * w / 10)
        print('widget_1_w', widget_1_w)
        widget_1_h = int(h - tabWidget_h - tabWidget_y -20)
        widget_1.setGeometry(widget_1_x, widget_1_y, widget_1_w, widget_1_h)  #
        #图片显示窗口变化
        horizontalWidget_2_x = 0
        horizontalWidget_2_y = 0
        horizontalWidget_2_w = tabWidget_w
        horizontalWidget_2_h = tabWidget_h
        horizontalWidget_2.setGeometry(horizontalWidget_2_x, horizontalWidget_2_y, horizontalWidget_2_w,
                                       horizontalWidget_2_h)
        if pixmap3:  # 防止为空
            pixmap3 = pixmap3.scaled(label_8.width(), label_8.height())
            label_8.setPixmap(pixmap3)
        if pixmap4:
            pixmap4 = pixmap4.scaled(label_7.width(), label_7.height())
            label_7.setPixmap(pixmap4)
        #
        #视频显示窗口变化
        label_6_x = 0
        label_6_y = 0
        label_6_w = tabWidget_w
        label_6_h = tabWidget_h
        label_6.setGeometry(label_6_x, label_6_y, label_6_w, label_6_h)
        if pixmap2:
            pixmap2 = pixmap2.scaled(label_6.width(), label_6.height())
            label_6.setPixmap(pixmap2)
        #
        #文件夹显示窗口变化
        horizontalWidget_x = 0
        horizontalWidget_y = 0
        horizontalWidget_w = tabWidget_w
        horizontalWidget_h = tabWidget_h
        horizontalWidget.setGeometry(horizontalWidget_x, horizontalWidget_y, horizontalWidget_w,
                                       horizontalWidget_h)
        if pixmap0:  # 防止为空
            pixmap0 = pixmap0.scaled(label_3.width(), label_3.height())
            label_3.setPixmap(pixmap0)
        if pixmap1:
            pixmap1 = pixmap1.scaled(label_5.width(), label_5.height())
            label_5.setPixmap(pixmap1)
        #
        #更多设置窗口变化
        gridWidget_x = int(tabWidget_w*0.2)
        gridWidget_y = int(tabWidget_h*0.1)
        gridWidget_w = int(tabWidget_w*0.6)
        gridWidget_h = int(tabWidget_h*0.8)
        gridWidget.setGeometry(gridWidget_x,gridWidget_y,gridWidget_w,gridWidget_h)
        #
        #登录界面窗口变化
        gridWidget_2_x = int(w * 0.3)
        gridWidget_2_y = int(h * 0.6)
        gridWidget_2_w = int(w * 0.35)
        gridWidget_2_h = int(h * 0.3)
        gridWidget_2.setGeometry(gridWidget_2_x,gridWidget_2_y,gridWidget_2_w,gridWidget_2_h)
def progressBar_work(value):
    progressBar.setValue(value)
    pass
def error_message():
    QMessageBox.information(None, '错误', '检测模式和待检测数据类型不匹配', QMessageBox.StandardButton.Ok)
def init():
    global show_value, iou, conf, save_txt, save_conf, save_crop, exist_ok, stop, password_ok, user_ok, w, h
    global pixmap0, pixmap1, pixmap2, pixmap3, pixmap4  # 图片自适应
    password_ok = False
    user_ok = False
    w = 0
    h = 0
    iou_default = 0.7
    conf_default = 0.25
    save_txt_default = False
    save_conf_default = False
    save_crop_default = False
    exist_ok_default = True
    iou = iou_default
    conf = conf_default
    save_txt = save_txt_default
    save_conf = save_conf_default
    save_crop = save_crop_default
    exist_ok = exist_ok_default
    show_value = False
    pixmap0 = 0
    pixmap1 = 0
    pixmap2 = 0
    pixmap3 = 0
    pixmap4 = 0
    progressBar.reset()
class Ui_YOLOV8_SAR_Detection(object):
    def setupUi(self, YOLOV8_SAR_Detection):
        YOLOV8_SAR_Detection.setObjectName("YOLOV8_SAR_Detection")
        YOLOV8_SAR_Detection.resize(802, 620)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(YOLOV8_SAR_Detection.sizePolicy().hasHeightForWidth())
        YOLOV8_SAR_Detection.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(parent=YOLOV8_SAR_Detection)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(parent=self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(30, 9, 741, 401))
        self.tabWidget.setObjectName("tabWidget")
        self.tab_5 = QtWidgets.QWidget()
        self.tab_5.setObjectName("tab_5")
        self.horizontalWidget_2 = QtWidgets.QWidget(parent=self.tab_5)
        self.horizontalWidget_2.setGeometry(QtCore.QRect(10, 30, 711, 351))
        self.horizontalWidget_2.setObjectName("horizontalWidget_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalWidget_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_8 = QtWidgets.QLabel(parent=self.horizontalWidget_2)
        self.label_8.setText("")
        self.label_8.setObjectName("label_8")
        self.horizontalLayout_2.addWidget(self.label_8)
        self.label_7 = QtWidgets.QLabel(parent=self.horizontalWidget_2)
        self.label_7.setText("")
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_2.addWidget(self.label_7)
        self.tabWidget.addTab(self.tab_5, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.label_6 = QtWidgets.QLabel(parent=self.tab_4)
        self.label_6.setGeometry(QtCore.QRect(60, 50, 531, 141))
        self.label_6.setText("")
        self.label_6.setObjectName("label_6")
        self.tabWidget.addTab(self.tab_4, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.horizontalWidget = QtWidgets.QWidget(parent=self.tab_3)
        self.horizontalWidget.setGeometry(QtCore.QRect(10, 30, 711, 351))
        self.horizontalWidget.setObjectName("horizontalWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_3 = QtWidgets.QLabel(parent=self.horizontalWidget)
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        self.horizontalLayout.addWidget(self.label_3)
        self.label_5 = QtWidgets.QLabel(parent=self.horizontalWidget)
        self.label_5.setText("")
        self.label_5.setObjectName("label_5")
        self.horizontalLayout.addWidget(self.label_5)
        self.tabWidget.addTab(self.tab_3, "")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.gridWidget = QtWidgets.QWidget(parent=self.tab)
        self.gridWidget.setGeometry(QtCore.QRect(150, 10, 471, 172))
        self.gridWidget.setObjectName("gridWidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.gridWidget)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_9 = QtWidgets.QLabel(parent=self.gridWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_9.sizePolicy().hasHeightForWidth())
        self.label_9.setSizePolicy(sizePolicy)
        self.label_9.setObjectName("label_9")
        self.gridLayout_2.addWidget(self.label_9, 0, 0, 1, 1)
        self.lineEdit_4 = QtWidgets.QLineEdit(parent=self.gridWidget)
        self.lineEdit_4.setCursorPosition(0)
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.gridLayout_2.addWidget(self.lineEdit_4, 0, 1, 1, 2)
        self.label_11 = QtWidgets.QLabel(parent=self.gridWidget)
        self.label_11.setObjectName("label_11")
        self.gridLayout_2.addWidget(self.label_11, 0, 3, 1, 1)
        self.label_10 = QtWidgets.QLabel(parent=self.gridWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_10.sizePolicy().hasHeightForWidth())
        self.label_10.setSizePolicy(sizePolicy)
        self.label_10.setObjectName("label_10")
        self.gridLayout_2.addWidget(self.label_10, 1, 0, 1, 1)
        self.lineEdit_5 = QtWidgets.QLineEdit(parent=self.gridWidget)
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.gridLayout_2.addWidget(self.lineEdit_5, 1, 1, 1, 2)
        self.label_12 = QtWidgets.QLabel(parent=self.gridWidget)
        self.label_12.setObjectName("label_12")
        self.gridLayout_2.addWidget(self.label_12, 1, 3, 1, 1)
        self.label_13 = QtWidgets.QLabel(parent=self.gridWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_13.sizePolicy().hasHeightForWidth())
        self.label_13.setSizePolicy(sizePolicy)
        self.label_13.setObjectName("label_13")
        self.gridLayout_2.addWidget(self.label_13, 2, 0, 1, 1)
        self.lineEdit_6 = QtWidgets.QLineEdit(parent=self.gridWidget)
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.gridLayout_2.addWidget(self.lineEdit_6, 2, 1, 1, 2)
        self.label_15 = QtWidgets.QLabel(parent=self.gridWidget)
        self.label_15.setObjectName("label_15")
        self.gridLayout_2.addWidget(self.label_15, 2, 3, 1, 1)
        self.label_16 = QtWidgets.QLabel(parent=self.gridWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_16.sizePolicy().hasHeightForWidth())
        self.label_16.setSizePolicy(sizePolicy)
        self.label_16.setObjectName("label_16")
        self.gridLayout_2.addWidget(self.label_16, 3, 0, 1, 1)
        self.lineEdit_7 = QtWidgets.QLineEdit(parent=self.gridWidget)
        self.lineEdit_7.setEchoMode(QtWidgets.QLineEdit.EchoMode.Normal)
        self.lineEdit_7.setObjectName("lineEdit_7")
        self.gridLayout_2.addWidget(self.lineEdit_7, 3, 1, 1, 2)
        self.label_17 = QtWidgets.QLabel(parent=self.gridWidget)
        self.label_17.setObjectName("label_17")
        self.gridLayout_2.addWidget(self.label_17, 3, 3, 1, 1)
        self.label_21 = QtWidgets.QLabel(parent=self.gridWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_21.sizePolicy().hasHeightForWidth())
        self.label_21.setSizePolicy(sizePolicy)
        self.label_21.setObjectName("label_21")
        self.gridLayout_2.addWidget(self.label_21, 4, 0, 1, 1)
        self.lineEdit_8 = QtWidgets.QLineEdit(parent=self.gridWidget)
        self.lineEdit_8.setEchoMode(QtWidgets.QLineEdit.EchoMode.Normal)
        self.lineEdit_8.setObjectName("lineEdit_8")
        self.gridLayout_2.addWidget(self.lineEdit_8, 4, 1, 1, 2)
        self.label_22 = QtWidgets.QLabel(parent=self.gridWidget)
        self.label_22.setObjectName("label_22")
        self.gridLayout_2.addWidget(self.label_22, 4, 3, 1, 1)
        self.label_18 = QtWidgets.QLabel(parent=self.gridWidget)
        self.label_18.setText("")
        self.label_18.setObjectName("label_18")
        self.gridLayout_2.addWidget(self.label_18, 5, 1, 1, 1)
        self.pushButton_7 = QtWidgets.QPushButton(parent=self.gridWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_7.sizePolicy().hasHeightForWidth())
        self.pushButton_7.setSizePolicy(sizePolicy)
        self.pushButton_7.setObjectName("pushButton_7")
        self.gridLayout_2.addWidget(self.pushButton_7, 5, 2, 1, 1)
        self.pushButton_8 = QtWidgets.QPushButton(parent=self.gridWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_8.sizePolicy().hasHeightForWidth())
        self.pushButton_8.setSizePolicy(sizePolicy)
        self.pushButton_8.setObjectName("pushButton_8")
        self.gridLayout_2.addWidget(self.pushButton_8, 5, 3, 1, 1)
        self.tabWidget.addTab(self.tab, "")
        self.widget = QtWidgets.QWidget(parent=self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(0, -1, 801, 581))
        self.widget.setObjectName("widget")
        self.label_20 = QtWidgets.QLabel(parent=self.widget)
        self.label_20.setGeometry(QtCore.QRect(0, 0, 801, 581))
        self.label_20.setBaseSize(QtCore.QSize(0, 0))
        self.label_20.setObjectName("label_20")
        self.gridWidget_2 = QtWidgets.QWidget(parent=self.widget)
        self.gridWidget_2.setGeometry(QtCore.QRect(310, 300, 175, 100))
        self.gridWidget_2.setObjectName("gridWidget_2")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.gridWidget_2)
        self.gridLayout_4.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.pushButton_9 = QtWidgets.QPushButton(parent=self.gridWidget_2)
        self.pushButton_9.setObjectName("pushButton_9")
        self.gridLayout_4.addWidget(self.pushButton_9, 2, 1, 1, 1)
        self.lineEdit_11 = QtWidgets.QLineEdit(parent=self.gridWidget_2)
        self.lineEdit_11.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self.lineEdit_11.setObjectName("lineEdit_11")
        self.gridLayout_4.addWidget(self.lineEdit_11, 1, 1, 1, 1)
        self.label_24 = QtWidgets.QLabel(parent=self.gridWidget_2)
        self.label_24.setObjectName("label_24")
        self.gridLayout_4.addWidget(self.label_24, 1, 0, 1, 1)
        self.lineEdit_10 = QtWidgets.QLineEdit(parent=self.gridWidget_2)
        self.lineEdit_10.setObjectName("lineEdit_10")
        self.gridLayout_4.addWidget(self.lineEdit_10, 0, 1, 1, 1)
        self.label_23 = QtWidgets.QLabel(parent=self.gridWidget_2)
        self.label_23.setObjectName("label_23")
        self.gridLayout_4.addWidget(self.label_23, 0, 0, 1, 1)
        self.widget_1 = QtWidgets.QWidget(parent=self.centralwidget)
        self.widget_1.setGeometry(QtCore.QRect(170, 440, 510, 132))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_1.sizePolicy().hasHeightForWidth())
        self.widget_1.setSizePolicy(sizePolicy)
        self.widget_1.setBaseSize(QtCore.QSize(0, 0))
        self.widget_1.setObjectName("widget_1")
        self.gridLayout = QtWidgets.QGridLayout(self.widget_1)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(parent=self.widget_1)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.lineEdit = QtWidgets.QLineEdit(parent=self.widget_1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit.sizePolicy().hasHeightForWidth())
        self.lineEdit.setSizePolicy(sizePolicy)
        self.lineEdit.setInputMask("")
        self.lineEdit.setText("")
        self.lineEdit.setPlaceholderText("")
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout.addWidget(self.lineEdit, 0, 1, 1, 3)
        self.pushButton_5 = QtWidgets.QPushButton(parent=self.widget_1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_5.sizePolicy().hasHeightForWidth())
        self.pushButton_5.setSizePolicy(sizePolicy)
        self.pushButton_5.setBaseSize(QtCore.QSize(200, 100))
        self.pushButton_5.setObjectName("pushButton_5")
        self.gridLayout.addWidget(self.pushButton_5, 0, 4, 1, 2)
        self.label_2 = QtWidgets.QLabel(parent=self.widget_1)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.lineEdit_2 = QtWidgets.QLineEdit(parent=self.widget_1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_2.sizePolicy().hasHeightForWidth())
        self.lineEdit_2.setSizePolicy(sizePolicy)
        self.lineEdit_2.setPlaceholderText("")
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.gridLayout.addWidget(self.lineEdit_2, 1, 1, 1, 3)
        self.pushButton_3 = QtWidgets.QPushButton(parent=self.widget_1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_3.sizePolicy().hasHeightForWidth())
        self.pushButton_3.setSizePolicy(sizePolicy)
        self.pushButton_3.setObjectName("pushButton_3")
        self.gridLayout.addWidget(self.pushButton_3, 1, 4, 1, 2)
        self.label_4 = QtWidgets.QLabel(parent=self.widget_1)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 2, 0, 1, 1)
        self.lineEdit_3 = QtWidgets.QLineEdit(parent=self.widget_1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_3.sizePolicy().hasHeightForWidth())
        self.lineEdit_3.setSizePolicy(sizePolicy)
        self.lineEdit_3.setInputMask("")
        self.lineEdit_3.setText("")
        self.lineEdit_3.setPlaceholderText("")
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.gridLayout.addWidget(self.lineEdit_3, 2, 1, 1, 3)
        self.pushButton_4 = QtWidgets.QPushButton(parent=self.widget_1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_4.sizePolicy().hasHeightForWidth())
        self.pushButton_4.setSizePolicy(sizePolicy)
        self.pushButton_4.setObjectName("pushButton_4")
        self.gridLayout.addWidget(self.pushButton_4, 2, 4, 1, 2)
        self.pushButton = QtWidgets.QPushButton(parent=self.widget_1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton.sizePolicy().hasHeightForWidth())
        self.pushButton.setSizePolicy(sizePolicy)
        self.pushButton.setObjectName("pushButton")
        self.gridLayout.addWidget(self.pushButton, 3, 1, 1, 1)
        self.pushButton_2 = QtWidgets.QPushButton(parent=self.widget_1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_2.sizePolicy().hasHeightForWidth())
        self.pushButton_2.setSizePolicy(sizePolicy)
        self.pushButton_2.setObjectName("pushButton_2")
        self.gridLayout.addWidget(self.pushButton_2, 3, 2, 1, 1)
        self.radioButton = QtWidgets.QRadioButton(parent=self.widget_1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.radioButton.sizePolicy().hasHeightForWidth())
        self.radioButton.setSizePolicy(sizePolicy)
        self.radioButton.setObjectName("radioButton")
        self.gridLayout.addWidget(self.radioButton, 3, 3, 1, 1)
        self.label_14 = QtWidgets.QLabel(parent=self.widget_1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_14.sizePolicy().hasHeightForWidth())
        self.label_14.setSizePolicy(sizePolicy)
        self.label_14.setObjectName("label_14")
        self.gridLayout.addWidget(self.label_14, 3, 4, 1, 1)
        self.progressBar = QtWidgets.QProgressBar(parent=self.widget_1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.progressBar.sizePolicy().hasHeightForWidth())
        self.progressBar.setSizePolicy(sizePolicy)
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        self.gridLayout.addWidget(self.progressBar, 3, 5, 1, 1)
        self.tabWidget.raise_()
        self.widget_1.raise_()
        self.widget.raise_()
        YOLOV8_SAR_Detection.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=YOLOV8_SAR_Detection)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 802, 22))
        self.menubar.setObjectName("menubar")
        self.menuSAR = QtWidgets.QMenu(parent=self.menubar)
        self.menuSAR.setTitle("")
        self.menuSAR.setObjectName("menuSAR")
        YOLOV8_SAR_Detection.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=YOLOV8_SAR_Detection)
        self.statusbar.setObjectName("statusbar")
        YOLOV8_SAR_Detection.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menuSAR.menuAction())
        self.retranslateUi(YOLOV8_SAR_Detection)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(YOLOV8_SAR_Detection)
    def retranslateUi(self, YOLOV8_SAR_Detection):
        _translate = QtCore.QCoreApplication.translate
        YOLOV8_SAR_Detection.setWindowTitle(_translate("YOLOV8_SAR_Detection", "MainWindow"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_5), _translate("YOLOV8_SAR_Detection", "图片"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), _translate("YOLOV8_SAR_Detection", "视频"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("YOLOV8_SAR_Detection", "文件夹"))
        self.label_9.setText(_translate("YOLOV8_SAR_Detection", "iou"))
        self.lineEdit_4.setPlaceholderText(_translate("YOLOV8_SAR_Detection", "默认0.7,取值范围：0~1"))
        self.label_11.setText(_translate("YOLOV8_SAR_Detection", "NMS 的交集并集 （IoU） 阈值"))
        self.label_10.setText(_translate("YOLOV8_SAR_Detection", "conf"))
        self.lineEdit_5.setPlaceholderText(_translate("YOLOV8_SAR_Detection", "默认0.25,取值范围：0~1"))
        self.label_12.setText(_translate("YOLOV8_SAR_Detection", "检测的对象置信度阈值"))
        self.label_13.setText(_translate("YOLOV8_SAR_Detection", "save_txt"))
        self.lineEdit_6.setPlaceholderText(_translate("YOLOV8_SAR_Detection", "默认False，取值范围True/False"))
        self.label_15.setText(_translate("YOLOV8_SAR_Detection", "将结果（类别，位置）另存为 .txt 文件"))
        self.label_16.setText(_translate("YOLOV8_SAR_Detection", "save_conf"))
        self.lineEdit_7.setPlaceholderText(_translate("YOLOV8_SAR_Detection", "默认False，取值范围True/False"))
        self.label_17.setText(_translate("YOLOV8_SAR_Detection", "将置信度分数保存入结果"))
        self.label_21.setText(_translate("YOLOV8_SAR_Detection", "save_crop"))
        self.lineEdit_8.setPlaceholderText(_translate("YOLOV8_SAR_Detection", "默认False，取值范围True/False"))
        self.label_22.setText(_translate("YOLOV8_SAR_Detection", "保存带有结果的裁剪图像"))
        self.pushButton_7.setText(_translate("YOLOV8_SAR_Detection", "确定"))
        self.pushButton_8.setText(_translate("YOLOV8_SAR_Detection", "恢复默认设置"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("YOLOV8_SAR_Detection", "更多设置"))
        self.label_20.setText(_translate("YOLOV8_SAR_Detection", "TextLabel"))
        self.pushButton_9.setText(_translate("YOLOV8_SAR_Detection", "登录"))
        self.label_24.setText(_translate("YOLOV8_SAR_Detection", "密码："))
        self.label_23.setText(_translate("YOLOV8_SAR_Detection", "账号："))
        self.label.setText(_translate("YOLOV8_SAR_Detection", "模型选择"))
        self.pushButton_5.setText(_translate("YOLOV8_SAR_Detection", "打开文件"))
        self.label_2.setText(_translate("YOLOV8_SAR_Detection", "待检测数据选择"))
        self.pushButton_3.setText(_translate("YOLOV8_SAR_Detection", "打开文件"))
        self.label_4.setText(_translate("YOLOV8_SAR_Detection", "检测结果保存路径"))
        self.pushButton_4.setText(_translate("YOLOV8_SAR_Detection", "打开文件"))
        self.pushButton.setText(_translate("YOLOV8_SAR_Detection", "检测"))
        self.pushButton_2.setText(_translate("YOLOV8_SAR_Detection", "停止检测"))
        self.radioButton.setText(_translate("YOLOV8_SAR_Detection", "显示            "))
        self.label_14.setText(_translate("YOLOV8_SAR_Detection", "进度："))
if __name__ == '__main__':
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_YOLOV8_SAR_Detection()  # 这是类函数的名称
    ui.setupUi(MainWindow)  # 运行类函数里的setupUi
    ui.retranslateUi(MainWindow)
    MainWindow.setWindowTitle("YOLOv8SAR船舶图像识别")
    centralwidget: QWidget = ui.centralwidget
    lineEdit: QLineEdit = ui.lineEdit
    lineEdit_2: QLineEdit = ui.lineEdit_2
    lineEdit_3: QLineEdit = ui.lineEdit_3
    lineEdit_4: QLineEdit = ui.lineEdit_4
    lineEdit_5: QLineEdit = ui.lineEdit_5
    lineEdit_6: QLineEdit = ui.lineEdit_6
    lineEdit_7: QLineEdit = ui.lineEdit_7
    lineEdit_8: QLineEdit = ui.lineEdit_8
    lineEdit_10: QLineEdit = ui.lineEdit_10
    lineEdit_11: QLineEdit = ui.lineEdit_11
    pushButton: QPushButton = ui.pushButton
    pushButton_2: QPushButton = ui.pushButton_2
    pushButton_3: QPushButton = ui.pushButton_3
    pushButton_4: QPushButton = ui.pushButton_4
    pushButton_5: QPushButton = ui.pushButton_5
    pushButton_7: QPushButton = ui.pushButton_7
    pushButton_8: QPushButton = ui.pushButton_8
    pushButton_9: QPushButton = ui.pushButton_9
    label_3: QLabel = ui.label_3
    label_5: QLabel = ui.label_5
    label_6: QLabel = ui.label_6
    label_7: QLabel = ui.label_7
    label_8: QLabel = ui.label_8
    label_20: QLabel = ui.label_20
    tabWidget: QTabWidget = ui.tabWidget  # 标签页框
    widget: QWidget = ui.widget  # 登录界面
    widget_1: QWidget = ui.widget_1  # 打开文件，预测等一系列按钮，操作框
    horizontalWidget: QWidget = ui.horizontalWidget  # 文件夹图片显示框
    horizontalWidget_2: QWidget = ui.horizontalWidget_2  # 图片显示框
    gridWidget: QWidget = ui.gridWidget  # 更多设置显示框
    gridWidget_2: QWidget = ui.gridWidget_2  # 登录显示框
    progressBar: QProgressBar = ui.progressBar  # 进度条
    init()
    timer = QTimer(MainWindow)
    timer.start(300)
    timer.timeout.connect(lambda: change_size())  # 每200毫秒查询窗口是否变化
    radioButton: QRadioButton = ui.radioButton
    pushButton.clicked.connect(lambda: pre(lineEdit.text(), lineEdit_2.text(), lineEdit_3.text()))
    pushButton_2.clicked.connect(lambda: stop_main())
    pushButton_3.clicked.connect(lambda: open_folder_dialog(3))
    pushButton_4.clicked.connect(lambda: open_folder_dialog(4))
    pushButton_5.clicked.connect(lambda: open_folder_dialog(5))
    pushButton_7.clicked.connect(lambda: update_configuration(0))
    pushButton_8.clicked.connect(lambda: update_configuration(1))
    pushButton_9.clicked.connect(lambda: log_in(lineEdit_10.text(), lineEdit_11.text()))
    radioButton.pressed.connect(lambda: show())
    background()
    MainWindow.show()  # 显示窗口
    sys.exit(app.exec())
