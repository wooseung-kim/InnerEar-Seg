# InnerEar-Seg
Official repository for the paper "Deep Learning-Based Inner Ear Subregion Segmentation in 3D T2-Weighted MRI Using Label-Preserving Data Augmentation."

---

## Installation

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py --train-dir [path/to/your/train/data/dir] --valid-dir [path/to/your/valid/data/dir] --log-dir [path/to/log/dir]
```

## Test

```bash
python test.py --test-dir [path/to/your/test/data/dir] --log-dir [path/to/log/dir]
```
