# YOLOv4-tiny + SORT

Installation
------------
1. Install "pycuda".  Note that the installation script resides in the "ssd" folder.

   ```shell
   $ cd ${PATH_PROJECT}/Detection/ssd
   $ ./install_pycuda.sh
   ```

2. Install **version "1.4.1" (not the latest version)** of python3 **"onnx"** module.  Note that the "onnx" module would depend on "protobuf" as stated in the [Prerequisite](#prerequisite) section.  Reference: [information provided by NVIDIA](https://devtalk.nvidia.com/default/topic/1052153/jetson-nano/tensorrt-backend-for-onnx-on-jetson-nano/post/5347666/#5347666).

   ```shell
   $ sudo pip3 install onnx==1.4.1
   ```

3. Go to the "plugins/" subdirectory and build the "yolo_layer" plugin.  When done, a "libyolo_layer.so" would be generated.

   ```shell
   $ cd ${PATH_PROJECT}/Detection/plugins
   $ make
   ```

4. Download the pre-trained yolov4-tiny COCO models and convert the targeted model to ONNX and then to TensorRT engine. 
   
   ```shell
   $ cd ${PATH_PROJECT}/Detection/yolo
   $ ./download_yolo.sh
   $ python3 yolo_to_onnx.py -m yolov4-tiny-416
   $ python3 onnx_to_tensorrt.py -m yolov4-tiny-416
   ```

   The last step ("onnx_to_tensorrt.py") takes a little bit more than half an hour to complete on my Jetson Nano DevKit.  When that is done, the optimized TensorRT engine would be saved as "yolov4-tiny-416.trt".

5. Run the test.py
   ```shell
   $ cd ${PATH_PROJECT}/Detection/
   $ python3 test.py
   ```

