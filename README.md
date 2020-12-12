# Example

This an example model repo for abandon detect model.

## Usage

*[**yolov5.weights下载**](https://pan.baidu.com/s/1UM2-h7JMR0llglHej6hEXg)
提取码（ 122w ）

python demo.py  --source images/video1.avi --output images/output/out.avi

注意！：  

#### 参数说明

相关配置参数在
*[**abandon_config.py**](https://github.com/dmuqlzhang/abandon_detection_yolov5/blob/main_github/abandon_config.py)
下
包括遗留物类别、时间阈值、iou阈值等

--output 测试视频输出路径


#### 测试结果

![可视化img](images/output/out.gif)



