---
license: mit
base_model:
- timm/tf_efficientnetv2_s.in1k
pipeline_tag: image-classification
---
# SurfaceAI: models for road type, surface type and quality classification of street-level imagery

This repository provides the python code as well as the model files with weights and parameters of the SurfaceAI models.

The models are designed to infer the following information from street-level images:
- **Road type** (e.g., road, bike lane, cycleway, footway, (unspedified) path, no focus/no street)
- **Surface type** (asphalt, concrete, paving stones, sett, unpaved)
- **Surface quality**, where a dedicated model is used for each surface type

Currently, the models are based on Convolutional Neural Networks (CNNs) and there is a separate model for each task, which can be expanded to include other architectures in future versions. Road type und surface type are classification tasks, surface quality is considered as a regression.

**Example usage**:
The file `prediction_example.py` demonstrates how to process images to generate predictions. The model interface `ModelInterface` from file `Models.py` is initialized with a configuration dictionary including the desired model files. Only tasks for which a model is defined are performed. The prediction of images is performed by the method `batch_classifications`, which takes a list of PIL Images or NumPy Arrays. 

Using the example prediction file, the output of the classification method for the image ![example_image](example_images/IMG_20210226_172956.jpg) is a list

```
[
    'IMG_20210226_172956',
    '1_1_road__1_1_road_general',
    [3.061619645450264e-05, 0.9993541836738586, 1.4675654711027164e-05, 3.29997310473118e-05, 0.0003072938707191497, 5.346190690147523e-08, 0.00018766717403195798, 7.268250919878483e-05],
    'asphalt',
    [0.9866762757301331, 0.0002866282011382282, 0.00020359903282951564, 7.299235585378483e-05, 0.012760424986481667],
    'good',
    2.2281060218811035
]
```

with:
1. image id provided to the model
2. road type prediction
3. list of probability scores of all possible road types
4. surface type prediction
5. list of probability scores of all possible surface types
6. surface qualiy prediction
7. value of surface quality prediction

Note: When handling large numbers of images, it is the user’s responsibility to split the images into suitable batch sizes before passing them to the batch_classification method in order to avoid memory overflows. The optimal batch size depends on the available GPU or CPU memory capacity and should be tested accordingly.


**Folder structure**:
The model files are sorted in folders according to version, including a json file containing metadata of all models. The transformations used for training are recommended for prediction and are contained in the metadata file.
The current version is v1.

**Training Data**:
The surface type and quality models are trained on the dataset StreetSurfaceVis. See respective [paper](    
https://doi.org/10.48550/arXiv.2407.21454).

**Application**:
The models are used in SurfaceAI pipeline. See respective [paper](https://dl.acm.org/doi/10.1145/3681780.3697277) and [code](https://github.com/SurfaceAI/road_network_classification).


**Contact**:
This is part of the SurfaceAI project at the University of Applied Sciences, HTW Berlin.

surface-ai@htw-berlin.de

https://surfaceai.github.io/surfaceai/


**Citation**:
If you use these models please cite as:
```

@article{kapp_streetsurfacevis_2025,
	title = {{StreetSurfaceVis}: a dataset of crowdsourced street-level imagery annotated by road surface type and quality},
	author = {Kapp, Alexandra and Hoffmann, Edith and Weigmann, Esther and Mihaljević, Helena},
	volume = {12},
	issn = {2052-4463},
	url = {https://doi.org/10.1038/s41597-024-04295-9},
	doi = {10.1038/s41597-024-04295-9},
	number = {1},
	journal = {Scientific Data},
	month = jan,
	year = {2025},
	pages = {92},
}
```


**Funding**:
SurfaceAI is a mFund project funded by the Federal Ministry for Digital and Transportation Germany.
