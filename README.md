# Neural Beacon Placement

This code accompanies the paper *Jointly Optimizing Placement and Inference for Beacon-based Localization*.

Arxiv Link: [https://arxiv.org/abs/1703.08612](https://arxiv.org/abs/1703.08612)

Dependencies: Numpy, Tensorflow, and Matplotlib (for visualizations)


### Important Files:

**src/experiments** - This directory contains files defining the parameters for each experiment. Newly created experiment files should be placed here.

**src/config.py** - This file defines the paths used for saving data, model weights, and results.



### Evaluate a pretrained model:

We provide 6 pretrained models you can use to reproduce our results. Download the models [here](https://github.com/ayanc/NSP/releases/download/untagged-b778691ee67fe075aa38/pretrained_models.zip).

To evaluate a model, run the following commands:


```bash
unzip path_to_pretrained_models.zip
cd src
python gen_test_data.py maps/map1.txt #~200MB for each map
python eval_model.py anneal_map1
python gen_viz.py anneal_map1
```
Replace "anneal_map1" with another experiment name to evaluate other models.

Since the propagation model is noisy, your numbers may differ slightly from ours.


### Train a new model:

To train a new model, create a new experiment file in the src/experiments directory. Then, run the following commands:

```bash
cd src
python gen_train_data.py maps/map1.txt #~3GB for each map
python run.py exp_name #Replace "exp_name" with the name of your experiment
```

### Generate a new map:

To use a new map, convert the map to a .svg file. Then, run:
```bash
cd maps
python svg2txt.py path_to_svg 25 25 > path_to_map.txt # creates an evenly spaced grid of 25 x 25 beacon locations
```
To use the map, first generate train and test data:
```bash
python gen_train_data.py path_to_map.txt
python gen_test_data.py path_to_map.txt
```

Then, in an experiment file, set:
```python
MAPFILE = "path_to_map.txt"
```
