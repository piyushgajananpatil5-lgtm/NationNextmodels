# Dataset preparation

Download the aerial search-and-rescue sources into `data/raw/`. Check each dataset's license and terms before redistribution or commercial use.

## HERIDAL

Obtain HERIDAL from its official project/repository page, then unpack the downloaded archive under `data/raw/heridal`. The source distribution URL is intentionally not hard-coded because it is version- and access-dependent:

```bash
cd data/raw
mkdir -p heridal
# Download the licensed archive from the official HERIDAL release page, then:
unzip ~/Downloads/heridal.zip -d heridal
```

## SARD

Download SARD from the official dataset host and unpack it under `data/raw/sard`. The source distribution URL is intentionally not hard-coded because it is version- and access-dependent:

```bash
cd data/raw
mkdir -p sard
# Download the licensed archive from the official SARD release page, then:
unzip ~/Downloads/sard.zip -d sard
```

## VisDrone

The official VisDrone downloader uses URLs that can change. Download the train and validation archives from [VisDrone](https://github.com/VisDrone/VisDrone-Dataset), then unpack them under `data/raw/visdrone`:

The official repository maintains the current download links and terms. Open
the repository page, download `VisDrone2019-DET-train.zip` and
`VisDrone2019-DET-val.zip`, then run:

```bash
cd data/raw
mkdir -p visdrone
unzip ~/Downloads/VisDrone2019-DET-train.zip -d visdrone
unzip ~/Downloads/VisDrone2019-DET-val.zip -d visdrone
```

`prepare_dataset.py` accepts existing YOLO labels beside images, `annotations.csv` files with columns `image,class,x1,y1,x2,y2`, or native VisDrone annotations when `--visdrone` is supplied. VisDrone pedestrian categories become `person`; other categories are ignored because they are outside this model's SAR taxonomy.

```bash
python data/prepare_dataset.py --raw data/raw --output data/processed --val-fraction 0.2
# For native VisDrone annotations:
python data/prepare_dataset.py --raw data/raw/visdrone --output data/processed --visdrone --val-fraction 0.2
```

The result is `data/processed/data.yaml`, with `images/{train,val}` and `labels/{train,val}` directories.
