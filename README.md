# SS project 

## Project description
This is HW12 in the course Deep Learning for Sound Processing.

In this project SpEx+ ([paper](https://www.isca-speech.org/archive/pdfs/interspeech_2020/ge20_interspeech.pdf)) model was implemented for Speaker Separation task.

## Project structure
- **/hw_ss** - project scripts
- _install_dependencies.sh_ - script for dependencies installation
- _requirements.txt_ - Python requirements list
- _train.py_ - script to run train
- _test.py_ - script to run test

## Installation guide

It is strongly recommended to use new virtual environment for this project.

To install all required dependencies and final model run:
```shell
./install_dependencies.sh
```

## Reproduce results
To run train with _LibriSpeech_ _train-100_ dataset:
```shell
python -m train -c hw_asr/configs/train_config.json
```

To run test inference with custom dataset:
```shell
python test.py \
   -c hw_ss/configs/test_config.json \
   -r best_model/best_model.pth \
   -o test_result.json \
   -t PATH_TO_YOUR_DIR \
   -b 4
```


## Author
Dmitrii Uspenskii HSE AMI 4th year.
