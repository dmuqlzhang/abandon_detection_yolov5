# Example

This an example model repo for abandon detect model.

## Usage

*[**yolov5.weights下载**](https://github-production-release-asset-2e65be.s3.amazonaws.com/264818686/7842dc00-19fd-11eb-8551-bf6aa5418b96?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20201212%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20201212T065504Z&X-Amz-Expires=300&X-Amz-Signature=57fe96173f0359c155b306f732751d7b25fdd79539b2cd615b16684ddd64c7b5&X-Amz-SignedHeaders=host&actor_id=47552281&key_id=0&repo_id=264818686&response-content-disposition=attachment%3B%20filename%3Dyolov5x.pt&response-content-type=application%2Foctet-stream)

python demo.py  --source images/video1.avi --output images/output/out.avi

注意！：  

#### 参数说明

相关配置参数在
*[**abandon_config.py**](https://github.com/dmuqlzhang/abandon_detection_yolov5/blob/master/README_ZN.md)
下
包括遗留物类别、时间阈值、iou阈值等

--output 测试视频输出路径


#### 测试结果

![可视化img](images/output/out.gif)



