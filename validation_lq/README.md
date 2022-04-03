
#IJB-S Evaluation

The distribution of IJB-S dataset is managed by the authors. 
Request dataset from the dataset authors. 

http://biometrics.cse.msu.edu/Publications/Face/Kalkaetal_IJBSIARPPAJanusSurveillanceVideoBenchmark_BTAS2018.pdf

Preprocess the dataset by
1. loosely cropping the faces with ground truth bounding boxes and organize them as
```
For videos: 
<PID>/videos_<VideoID>_<FRAME>.jpg
EX)
1/videos_5004_21750.jpg
1/videos_5004_21751.jpg
...

For images:
<PID>/img_<FRAME>.jpg
EX)
1/img_101146.jpg
1/img_101147.jpg
...
```

2. Align the loose crop faces with face alignment tool such as MTCNN. And save it in the same structure.


4. Run
```
python validate_IJB_S.py --data_root $DATA_ROOT --model_name ir50
```