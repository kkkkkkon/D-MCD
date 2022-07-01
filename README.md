## Denoised Maximum Classifier Discrepancy for Source-Free Unsupervised Domain Adaptation (AAAI 2022)

This is a pytorch Denoised Maximum Classifier Discrepancy for Source-Free Unsupervised Domain Adaptation.

### Environment Requirements
- Python 3.7.0
- Pytorch 1.4.0
- torchvision 0.5.0
- matplotlib
- sklearn
- scipy
- numpy

The data folder should be structured as follows:

```
├── data/
│   ├── dataset name/     
|   |   ├── domain1/
|   |   ├── domain2/
|   |   ├── domain3/
|   |   ├── .../
│   └── 
├── trained_model/
│   ├── source/     
|   |   ├── dataset name1/
|   |   ├── dataset name2/
|   |   ├── dataset name3/
|   |   ├── .../
│   └── target/
|   |   ├── dataset name1/
|   |   ├── dataset name2/
|   |   ├── dataset name3/
|   |   ├── .../
│   └── final/
|   |   ├── dataset name1/
|   |   ├── dataset name2/
|   |   ├── dataset name3/
|   |   ├── .../
```

### Running on visda datasets
```
sh run_visda.sh > run_visda.txt 
sh run_visda.sh > run_visda.txt 
```

### Running on office-home datasets
```
sh run_office_home.sh > run_office_home.txt 
sh run_office_home.sh > run_office_home.txt
```

### Acknowledge
Codes are adapted from [BCDM](https://github.com/BIT-DA/BCDM.git), [MCD](https://github.com/mil-tokyo/MCD_DA.git) and [SE](https://github.com/Britefury/self-ensemble-visual-domain-adapt.git). We thank them for their excellent projects.


### Citation
If you find this code useful please consider citing
```
@inproceedings{DMCD,
title = {Denoised Maximum Classifier Discrepancy for Source-Free Unsupervised Domain Adaptation},
author = {Tong Chu and Yahao Liu and Jinhong Deng and Wen Li and Lixin Duan},
booktitle = {Thirty-Sixth AAAI Conference on Artificial Intelligence (AAAI-22)},    
year = {2022}
}
```
