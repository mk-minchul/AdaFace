
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


# TinyFace Evaluation

1. Download the evaluation split from https://qmul-tinyface.github.io/
   1. Extract the zip file under `<data_root>`. 
2. TinyFace Evaluation images have to be aligned and resized. 
   1. You may perform the alignment yourself with MTCNN or download by completing this form [link](https://forms.gle/Mz1LNrQwn1Bwjvo86).
   2. Do not re-distribute the aligned data. It is released for encouraging reproducibility of research. 
   But if it infringes the copy-right of the original tinyface dataset, the aligned version may be retracted.
   5. Extract the zip file under `<data_root>`
3. You may run evaluation with the example script below.

```
python validate_tinyface.py --data_root <data_root> --model_name ir101_webface4m
```