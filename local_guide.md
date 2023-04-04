assumes you have the folder ``datasets/CUB_200_2011``
as a folder here

run
to generate datafiles split
```
crop.py
image augmentation, at the moment we are not using it. just move the train set into the folder
img_aug.py

main.py
```
commands

accompanying bboxes are saved as:
``` bbox_height_start, bbox_height_end, bbox_width_start, bbox_width_end```

10 chosen images run using local analysis using all_local_analysis.sh

```


python global_analysis.py -modeldir ./saved_models/Resnet34/002 -model 15nopush0.1564.pth
```
