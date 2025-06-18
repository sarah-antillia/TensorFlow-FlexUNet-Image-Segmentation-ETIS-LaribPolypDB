<h2>TensorFlow-FlexUNet-Image-Segmentation-ETIS-LaribPolypDB (2025/06/18)</h2>

This is the second experiment of Image Segmentation for ETIS-LaribPolypDB 
 based on our TensorFlowFlexUNet (TensorFlow Flexible UNet Image Segmentation Model for Multiclass) 
and, <a href="https://drive.google.com/file/d/1uXhQ2lhfZwSasSmo3jdMRvWckHDz0U2G/view?usp=sharing">
Augmented-ETIS-PNG-LaribPolypDB-ImageMask-Dataset.zip</a>, 
which was derived by us from 
<a href="https://www.kaggle.com/datasets/nguyenvoquocduong/etis-laribpolypdb"><b>ETIS-LaribPolypDB</b></a> on Kaggle website.
<br>
<br>
Our Multiclass TensorFlowFlexUNet, which uses categorized masks, can also be applied to 
single-class image segmentation models. 
This is because it inherently treats the background as one category and your single-class mask data as 
a second category. In essence, your single-class segmentation model will operate with two categorized classes within our Multiclass UNet framework.
<br>
<br>
<hr>
<b>Actual Image Segmentation for Images of 512x512 pixels</b><br>
As shown below, the inferred masks look similar to the ground truth masks. <br>

<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/ETIS-LaribPolypDB/mini_test/images/8.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/ETIS-LaribPolypDB/mini_test/masks/8.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/ETIS-LaribPolypDB/mini_test_output/8.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/ETIS-LaribPolypDB/mini_test/images/86.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/ETIS-LaribPolypDB/mini_test/masks/86.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/ETIS-LaribPolypDB/mini_test_output/86.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/ETIS-LaribPolypDB/mini_test/images/135.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/ETIS-LaribPolypDB/mini_test/masks/135.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/ETIS-LaribPolypDB/mini_test_output/135.png" width="320" height="auto"></td>
</tr>
</table>

<hr>

<br>

<h3>1. Dataset Citation</h3>
The dataset used here has been taken from the following kaggle web-site:<br>
<a href="https://www.kaggle.com/datasets/nguyenvoquocduong/etis-laribpolypdb"><b>ETIS-LaribPolypDB</b></a>.
<br>
<br>
<b>License</b>: Unknown
<br>
<h3>
<a id="2">
2 ETIS-LaribPolypDB ImageMask Dataset
</a>
</h3>
 If you would like to train this ETIS-LaribPolypDB Segmentation model by yourself,
 please download the dataset from the google drive  
<a href="https://drive.google.com/file/d/1uXhQ2lhfZwSasSmo3jdMRvWckHDz0U2G/view?usp=sharing">
Augmented-ETIS-PNG-LaribPolypDB-ImageMask-Dataset.zip</a>
, expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be
<pre>
./dataset
└─ETIS-LaribPolypDB
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
 
<b>ETIS-LaribPolypDB Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/ETIS-LaribPolypDB/ETIS-LaribPolypDB_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is enough to use for a training set of our segmentation model.

<br>
<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/ETIS-LaribPolypDB/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/ETIS-LaribPolypDB/asset/train_masks_sample.png" width="1024" height="auto">
<br>

<h3>
3 Train TensorflowUNet Model
</h3>
 We have trained ETIS-LaribPolypDBTensorflowUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/ETIS-LaribPolypDB/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/ETIS-LaribPolypDBand run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorflowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters</b> and large <b>base_kernels</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorflowFlexUNet.py">TensorflowFlesUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
base_filters   = 16
base_kernels   = (9,9)
num_layers     = 8
num_classes    = 2

dilation       = (3,3)
</pre>

<b>Learning rate</b><br>
Defined a very small learning rate.  
<pre>
[model]
learning_rate  = 0.00001
</pre>

<b>Online augmentation</b><br>
Disabled our online augmentation.You may train this model by setting this generator parameter to True. 
<pre>
[model]
model         = "TensorflowUNet"
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>

<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>
<b>Mask RGB_map</b><br>
[mask]
<pre>
mask_datatype    = "categorized"
mask_file_format = ".png"
;ETIS-LaribPolypDB rgb color map dict for 1+1 classes.
;    Background:black, Polyp:white
rgb_map = {(0,0,0):0,(255, 255, 255):1, }
</pre>
<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer callback.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
epoch_changeinfer        = False
epoch_changeinfer_dir    = "./epoch_changeinfer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/ETIS-LaribPolypDB/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>

<b>Epoch_change_inference output at middlepoint (epoch 28,29,30)</b><br>
<img src="./projects/TensorFlowFlexUNet/ETIS-LaribPolypDB/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (epoch 58,59,60)</b><br>
<img src="./projects/TensorFlowFlexUNet/ETIS-LaribPolypDB/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>

In this experiment, the training process was stopped at epoch 60 by EarlyStopping callback.<br><br>
<img src="./projects/TensorFlowFlexUNet/ETIS-LaribPolypDB/asset/train_console_output_at_epoch_60.png" width="720" height="auto"><br>
<br>

<a href="./projects/TensorFlowFlexUNet/ETIS-LaribPolypDB/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/ETIS-LaribPolypDB/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/ETIS-LaribPolypDB/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/ETIS-LaribPolypDB/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/ETIS-LaribPolypDB</b> folder,<br>
and run the following bat file to evaluate TensorflowUNet model for ETIS-LaribPolypDB.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorflowFlexUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/ETIS-LaribPolypDB/asset/evaluate_console_output_at_epoch_60.png" width="720" height="auto">
<br><br>Image-Segmentation-ETIS-LaribPolypDB

<a href="./projects/TensorFlowFlexUNet/ETIS-LaribPolypDB/evaluation.csv">evaluation.csv</a><br>

The loss (bce_dice_loss) to this ETIS-LaribPolypDB/test was very low, and dice_coef very high as shown below.
<br>
<pre>
categorical_crossentropy,0.017
dice_coef_multiclass,0.9933
</pre>
<br>

<h3>
5 Inference
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/ETIS-LaribPolypDB</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for ETIS-LaribPolypDB.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorflowFlexUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/ETIS-LaribPolypDB/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/ETIS-LaribPolypDB/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/ETIS-LaribPolypDB/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks </b><br>

<table>
<tr>
<th>Image</th>b
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/ETIS-LaribPolypDB/mini_test/images/8.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/ETIS-LaribPolypDB/mini_test/masks/8.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/ETIS-LaribPolypDB/mini_test_output/8.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/ETIS-LaribPolypDB/mini_test/images/75.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/ETIS-LaribPolypDB/mini_test/masks/75.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/ETIS-LaribPolypDB/mini_test_output/75.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/ETIS-LaribPolypDB/mini_test/images/147.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/ETIS-LaribPolypDB/mini_test/masks/147.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/ETIS-LaribPolypDB/mini_test_output/147.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/ETIS-LaribPolypDB/mini_test/images/barrdistorted_1001_0.3_0.3_139.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/ETIS-LaribPolypDB/mini_test/masks/barrdistorted_1001_0.3_0.3_139.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/ETIS-LaribPolypDB/mini_test_output/barrdistorted_1001_0.3_0.3_139.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/ETIS-LaribPolypDB/mini_test/images/barrdistorted_1002_0.3_0.3_70.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/ETIS-LaribPolypDB/mini_test/masks/barrdistorted_1002_0.3_0.3_70.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/ETIS-LaribPolypDB/mini_test_output/barrdistorted_1002_0.3_0.3_70.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/ETIS-LaribPolypDB/mini_test/images/barrdistorted_1002_0.3_0.3_91.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/ETIS-LaribPolypDB/mini_test/masks/barrdistorted_1002_0.3_0.3_91.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/ETIS-LaribPolypDB/mini_test_output/barrdistorted_1002_0.3_0.3_91.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>

<h3>
References
</h3>

<b>1. Automated Detection of Colorectal Polyp Utilizing Deep Learning Methods With Explainable AI</b><br>
Faysal Ahamed, Rabiul Islam, Nahiduzzaman, Jawadul Karim, Mohamed Arselene Ayari, Amith Khandakar<br>

<a href="https://ieeexplore.ieee.org/document/10534764">https://ieeexplore.ieee.org/document/10534764</a><br>
<br>
<br>
<b>2. Tensorflow-Image-Segmentation-Pre-Augmented-ETIS-LaribPolypDB</b><br>
Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-Pre-Augmented-ETIS-LaribPolypDB">
https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-Pre-Augmented-ETIS-LaribPolypDB
</a>
<br>


