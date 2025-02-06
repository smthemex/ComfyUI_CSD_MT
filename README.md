# ComfyUI_CSD_MT
[CSD_MT](https://github.com/Snowfallingplum/CSD-MT) is a method about 'Content-Style Decoupling for Unsupervised Makeup Transfer without Generating Pseudo Ground Truth', you can use it in comfyUI.

# Tips
* This method need little space and VRAM(or CPU),all the models is about 200M(4 weights),and best quality size is 256*256
* 这个方法所用4个模型加起来不到200M,所需显存或CPU可以忽略不记,当然,图片输出最佳质量是256*256.

# 1. Installation

In the ./ComfyUI/custom_node directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_CSD_MT.git
```
---

# 2. Requirements  

```
pip install -r requirements.txt
```
* When install dlib get error,use wheel to install it,download wheel from this [address](https://github.com/z-mahmud22/Dlib_Windows_Python3.x)
* 如果按照dlib库失败，从此链接下载对应你python版本的轮子，用以下命令安装：
 
```
pip install dlib-19.24.1-cp311-cp311-win_amd64.whl
```

# 3.Model
* 3.1 download 2 checkpoints (79999_iter.pth,CSD_MT.pth) from [google](https://drive.google.com/drive/folders/1pvSgkpsb7k6Ph1_oCmFkMQZPgL7PaTO0) or [baidu](https://pan.baidu.com/share/init?surl=C7K4xk5W0X65yUQh41AmfQ) password:1d3e.从百度云或者Google下载三个模型
* 3.2 download 'resnet18-5c106cde.pth' from [here](https://download.pytorch.org/models/resnet18-5c106cde.pth) 从链接下载resnet18-5c106cde.pth模型，不要改名字.
```
--  ComfyUI/models/CSDMT
    |-- 79999_iter.pth
    |-- CSD_MT.pth
    |--resnet18-5c106cde.pth #没有也会自动下载
```
```
--  ComfyUI/custom_node/ComfyUI_CSD_MT/quick_start/faceutils/dlibutils
    |--lms.dat  # Already in the project, no need to download,已内置在插件中,不用下载
```

# 4.Example
![](https://github.com/smthemex/ComfyUI_CSD_MT/blob/main/example.png)


# 5.Citation
```
@inproceedings{sun2024content,
  title={Content-Style Decoupling for Unsupervised Makeup Transfer without Generating Pseudo Ground Truth},
  author={Sun, Zhaoyang and Xiong, Shengwu and Chen, Yaxiong and Rong, Yi}
  journal={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  year={2024}
}
```
