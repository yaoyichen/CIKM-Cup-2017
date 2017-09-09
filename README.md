# CIKM AnalytiCup 2017: Short-Term Precipitation Forecasting Based on Radar Reflectivity Images

---------


This repo describes the final solution of Team Marmot, 
who finished in 1<sup>st</sup> place in the [CIKM AnalytiCup 2017](https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100066.0.0.3f6e7d83jNRFTh&raceId=231596) – 
"Short-Term Quantitative Precipitation Forecasting Challenge"
sponsored by Shenzhen Meteorological Bureau and Alibaba Group. 
We provide an approach to reasonably predict 
the short-term precipitation at the target site in the future 1-2 
hours based on historical radar reflectivity images. 

---------


## Introduction
In the CIKM AnalytiCup 2017 challenge, contestants are provided with radar maps within 1.5 hours. Each radar map covers the radar reflectivity of the surrounding 101 km × 101 km areas of the target site. Radar maps are measured at different time spans, specifically 15 time spans with an interval of 6 minutes, and in 4 different heights from 0.5km to 3.5km with an interval of 1km. Therefore, 60 historical radar images are provided for one sample to predict the total precipitation amount on the ground in the future 1-2 hours for each target site.

The [dataset](https://github.com/yaoyichen/CIKM-Cup-2017/blob/master/DATA.md) have been collected and desensitized by the Shenzhen Meteorological Bureau, during a total time span of 3 years, in which the first two years data are used for training and the third year for testing. Our task here is to predict the exact precipitation amount with the objective to minimize the prediction error. The root mean square error (RMSE) is used to evaluate the predicting performance in the contest.
<p align="center"><img src="http://static.zybuluo.com/Jessy923/dmc8aal4i1k5mfsak1flsif9/data_format.jpg" width="650" height="200" alt="Data format" /></p>

---------



## Framework

The current solution regards cloud trajectory method based on velocity 
vector obtained by the SIFT (Scale-invariant feature transform) 
algorithm which matches descriptors in the adjacent time frames. 
Convolutional neural network is adopted with features including 
pinpoint local radar images, spatial-temporal descriptions of the 
cloud movement, and the global description of the cloud pattern.  

<div  align="center"> <img src="http://static.zybuluo.com/Jessy923/2x5adueuf0vggrhz0beq814j/flowchart.png" width="650" height="250" alt="Item-based filtering" /></div>


In the pre-processing stage, sub-regional images are connected
 by template matching and formed into large domain panorama. 
 Key points are detected and described by the SIFT algorithm, 
 rendering local descriptions of the cloud structures. 
 The SIFT descriptors are pair-matched between two adjacent time 
 frames to acquire the relative displacement in the time interval. 
 Then the velocity field could be derived from the relative displacement at each of key point. Resorting to Taylor frozen hypothesis [1], the trajectory that passes through the target site can be extrapolated. The features are generally classified into three categories. The local radar images (41 × 41 km) along the extrapolated trace can provide direct association between radar reflection and precipitation. The temporal and altitudinal vectors describe the evolution of radar reflective statistics along different time frames and radar observation heights. The cloud pattern vector depicts the cloud type in the whole image area (101 × 101 km), which is embedded as the histogram of reflective intensity and SIFT descriptor classes. Convolution neural network model is adopted and the architecture is shown in the Figure. Local images are fed into a 3-layer convolution neural net and each layer includes a 4 × 4 convolution kernel and a 2 × 2 max pooling kernel. Then the output images are flattened and concatenated with other two types of features, and passed to a 3-layer fully connected neural net with the precipitation required to be predicted at the output layer.

---------
## Detailed Solution report
- Check the pdf file For detailed Solution report.
- Or visit the __Chinese__ version of solution report contributed by Team member Jessy [CIKM 中文解题报告](https://github.com/Jessicamidi/CIKM-Cup-2017/blob/master/README.md)

---------


## Requirements
```
- python 2.7
- numpy 1.11.3
- opencv 3.1.0
- tensorflow 1.2.1
- sklearn 0.18.1
- networkx 1.11

```


---------


## Program structure

### step 1: Data transform and Image Matching
```
- step1.1_rawdata_rewrite.py
"Transform the data type from ascii to ubyte format (8 bits unsigned binary) 
and save to new files, which would reduce the data size to 1/3, and would 
save the data transforming time when read by the python"

- step1.2_space_match.py
"Spatial template matching of sub-images"

- step1.3_temporal_match.py
"Temporal template matching of sub-images"

- step1.4_testAB_stitch.py
"Stitch images by cross-search among testA and testB set"
```

### step 2:Local Descriptor and Trace Tracking
```
- step2.1_SIFT_descriptor.py
"Calculate the histogram of SIFT descriptors"

- step2.2_trajectory.py
"Calculate the extrapolated trajectory at each of the target site"
```
### step 3:Feature Generation
```
- step3.1_trajectory_image.py
"Generate local image feature at the surroudning area
of the extrapolation time stamp. "

- step3.2_temporal_spatial_vector.py
"Generate temporal and altitudinal vector"

- step3.3_general_description.py
"Generate global description of the cloud pattern"

- step3.4_flatten_noniamge_feature.py
"Package features in step3.2 and step3.3 and save as binary array"

- step3.5_patch.py
"Package features for neural network model and save as binary array"
```

### step 4:Training Models
```
- step4.1_cnn_simple.py
"Convolutional neural network training"

- step4.2_nn_patch.py
"Neural net training for samples without tracked local image"

- step4.3_nn.py
"neural network model"

- step4.4_gbdt.py
"GBDT(gradient boosting decision tree model)"

- step4.5_submit.py
"Model ensemble and submit"
```
### TOOLS
```
- CIKM_TOOLS.py
"import libs and commonly used functions"
```

---------



### References
-	Taylor G. I. 1938. Production and dissipation of vorticity in a turbulent fluid. Proceedings of the Royal Society of London. Series A, Mathematical and Physical Sciences, 15-23.

-	Rosten, E., & Drummond, T. 2006. Machine learning for high-speed corner detection. Computer Vision–ECCV 2006, 430-443.

-	Lowe, D. G. (2004). Distinctive image features from scale-invariant key points. International journal of computer vision, 60(2), 91-110.

