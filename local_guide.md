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
```
python local_analysis.py -modeldir ./saved_models/Resnet34/002 -model 15nopush0.1564.pth -imgdir ./local_analysis/Painted_Bunting_Class15_0081/ -img Painted_Bunting_0081_15230.png -imgclass 15


python global_analysis.py -modeldir ./saved_models/Resnet34/002 -model 15nopush0.1564.pth
```
