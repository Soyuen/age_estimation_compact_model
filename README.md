## Age estimation compact model
 <img src="https://github.com/Soyuen/age_estimation_compact_model/blob/main/image/output.jpg" width = "393" height = "393" alt="output" align=center />
## Requirements
* Anaconda
* Python 3.7
* [Packages](https://github.com/Soyuen/age_estimation_compact_model/blob/main/packages.txt)

### Install packages
```
pip install -r packages.txt
```
## Procedure
### Data processcing
* Download [IMDB](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar) and [WIKI](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar) datasets.(https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)
* Extract files to "./data"
* Run the code below for data processing

```
cd ./data
python imdbwiki_filter.py
python imdbwiki_pre.py
```
### Training  model
Training model with 90 epochs.The batch size is respectively 128 on Imdb dataset and 50 on Wiki dataset.(the same setting with SSR-Net)
```
cd ./training
python train.py --input ../data/imdb.npz --db imdb
python train.py --input ../data/wiki.npz --db wiki  --batch_size 50
```

Measure the execution time on [GEFORCE RTX 2070](https://www.nvidia.com/en-us/geforce/graphics-cards/rtx-2070/).  
Because the official model of C3AE is unavailable, cannot perform the following tasks.


|  Model  | Imdb(MAE)  | Wiki(MAE)  | Parameters |Model Size|Execution time|                        |
|---------|:----------:|:----------:|:----------:|:--------:|:------------:|:-----------------------|
|SSR-Net  |   6.94     |    6.76    |   40915    |  319KB   |  2.9 ms      |original                |
|SSR-Net  |   6.41     |    7.01    |   40915    |  319KB   |   --         |new data proccessing    |
|C3AE     |   6.57     |    6.44    |   36345    |  198KB   |   --         |Plain model(from papper)|
|AEC_model|   6.28     |    6.21    |   20014    |  184KB   |  2.3 ms      |our model               |
 
 <img src="https://github.com/Soyuen/age_estimation_compact_model/blob/main/image/image.jpg" width = "507" height = "273" alt="result" align=center />
## Reference
C3AE : https://arxiv.org/pdf/1904.05059.pdf  
SSR-net : https://github.com/shamangary/SSR-Net
