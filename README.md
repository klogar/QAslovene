# Learning Question Answering in Slovene Language

### Katja Logar
##### Master's Thesis, Faculty of Computer and Information Science, University of Ljubljana

# Table of Contents
1. [Installation](#Installation)
2. [Usage](#Usage)
3. [Datasets](#Datasets)

## Installation

### Dependencies 
- Python 3
- [PyTorch](https://pytorch.org/) (currently tested on version 1.10.1)
- [Transformers](https://github.com/huggingface/transformers) (currently tested on version 4.19.0.dev0)

### Quick installation
```bash
pip install -r requirements.txt
```

## Usage

All scripts use ```run_qa.py```, which is a modified ```run_summarization.py``` script, provided by [Huggigface](https://github.com/huggingface/transformers).

### New parameters
We added some new parameters that are not present in the original script.
* ```--datasets```: list of all datasets that should be included for train/test/validation
* ```--datasets_path```: path to folder where datasets are present
* ```--lowercase```: whether all text should be cast to lowercase
* ```--filter_no_answer```: whether all examples which cannot be answered (mostly SQuAD 2.0 and some in MultiRC) should be skipped

#### Model Training
```bash
./run_train.sh
```

#### Generate predictions for each checkpoint for validation dataset
```bash
./run_predict_checkpoints.sh
```

#### Generate prediction for specific checkpoint
```bash
./run_predict.sh
```



#### Evaluation of predictions
Use ```evaluate.py```. You should set variable model (e.g. "slounifiedqa"). If you need to select best checkpoint from validation dataset, set checkpoint to "all" and kind to "val". If you are evaluating on test dataset, set checkpoint to selected best checkpoint and kind to "test_answered".

## Datasets
Encoded slovene datasets (BoolQ, MultiRC, COPA, SQuAD 2.0 and MCTest) that were used to obtain results are available [here](https://drive.google.com/file/d/1thAOseosns4qr_JsjhB0Wx1dvitrK3BB/view?usp=sharing). In addition to ```input``` and ```output``` column, there is also a ```type``` column that tells you whether that particular example was human translated (HT), machine translated (MT - this refers to all datasets that were previously machine translated and to internal neural translator that was used for MCTest dataset) or translated by DeepL (DEEPL). For MultiRC and SQuAD 2.0 which have multiple correct answers, the answers are in ```answers``` folder.

Translated MCTest dataset is available in two versions: translated by internal neural translator [here](https://drive.google.com/file/d/1ly02cPiPwiHv4tlhXXs8BW1sHcWqXSAj/view?usp=sharing) and translated by DeepL [here](https://drive.google.com/file/d/1r6Bil_B59BlNrVtkPx71forFe5ceV36f/view?usp=sharing).
