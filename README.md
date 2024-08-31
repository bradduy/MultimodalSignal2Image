# MultimodalSignal2Image

This 🔥 leverages the comprehensive EPHNOGRAM dataset, a rich repository of simultaneous electrocardiogram (ECG) and phonocardiogram (PCG) recordings, supplemented with environmental audio noise data. 

Our 🔥 focuses on developing and applying multimodal 🛜signal-to-image🌄 methods to the [EPHNOGRAM](https://physionet.org/content/ephnogram/1.0.0/) dataset. By converting these complex signals into image representations

Core functionality:

- ```preprocessing.py```: to pre-process dataset from (SIGNAL) 1-dimensional data to (IMAGE) 3-D dimensional ones by using matplotlib.

- ```model.py```: contain Computer Vision models (E.g: AlexNet).

- ```train_test.py```: to train/test generated (IMAGE) 3-D dimensional data with Computer Vision Classification models.
# Supported Models:
- CNN-based models: ```AlexNet```
- Transformer-based models: ```Vision Transformer (ViT)```
# Installation:
    pip install -r requirements.txt

# 📖 Tutorial:

1. Download [EPHNOGRAM](https://physionet.org/content/ephnogram/1.0.0/) dataset from official link, and unzip (if any).
2. Replace ```dataset_dir``` in ```preprocessing.py``` by the EPHNOGRAM downloaded path above. 
3. Run ```train_test.py``` to train models on signal-based image dataset for the 1st time.
4. [OPTIONAL] Run ```train_test.py``` to test training models on signal-based image dataset for the > 1st time.

## 🙋🏻‍♂️ Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.


## 🏆 Contributors

<a href="https://github.com/bradduy">
  <img src="https://avatars.githubusercontent.com/u/33892919?v=4" style="border-radius: 50%; width: 50px; height: 50px; object-fit: cover;" alt="LLMChainFusion contributors">
</a>
