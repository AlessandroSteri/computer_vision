# computer_vision

## Dataset
The dataset is available [here](https://drive.google.com/file/d/1v5HZtSFF0FH-5mr5sHHjabI51lDJFZjt/view?usp=sharing).
Flick30k + Keras and its [VGG16 pretrained models](https://github.com/fchollet/deep-learning-models) or download [feature files](https://drive.google.com/file/d/1_PteTR8vHF8kC9x1LYnW0b1q9A2ggAz3/view?usp=sharing)
[Glove embeddings](http://nlp.stanford.edu/data/glove.840B.300d.zip)

### Training
Obtain the image features by either extracting using Keras and Flickr30k dataset or download the features file (git-lfs)
   - To extract image features by yourself, specify location of Flickr30k dataset in [image_utils.py](bimpm/image_utils.py)
        and run ``` python image_utils.py```
2. Specify location of the embedding in the config file
3. ```python main.py --config_file=file_config_name_here.config```
