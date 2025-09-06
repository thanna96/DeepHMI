# Improving the Spatial Resolution of SDO/HMI Transverse and Line-of-Sight Magnetograms Using GST/NIRIS Data with Machine Learning


## Author
Chunhui Xu, Yan Xu, Jason T. L. Wang, Qin Li, Haimin Wang

**Affiliation:** Institute for Space Weather Sciences, New Jersey
Institute of Technology

## Abstract
High-resolution magnetograms are crucial for studying solar flare dynamics because they enable the precise tracking of magnetic structures and rapid field changes. The Helioseismic and Magnetic Imager on board the Solar Dynamics Observatory (SDO/HMI) has been an essential provider of vector magnetograms. However, the spatial resolution of the HMI magnetograms is limited and hence is not able to capture the fine structures that are essential for understanding flare precursors. The Near InfraRed Imaging Spectropolarimeter on the 1.6 m Goode Solar Telescope (GST/NIRIS) at Big Bear Solar Observatory provides a better spatial resolution and is therefore more suitable to track the fine magnetic features and their connection to flare precursors. We propose **DeepHMI**, a machine-learning method for solar image super-resolution, to enhance the transverse and line-of-sight magnetograms of solar active regions (ARs) collected by SDO/HMI to better capture the fine-scale magnetic structures. The enhanced HMI magnetograms can also be used to study spicules, sunspot light bridges and magnetic outbreaks, for which high-resolution data are essential.DeepHMI employs a conditional diffusion model that is trained using ground-truth images obtained by inversion analysis of Stokes measurements collected by GST/NIRIS. Our experiments show that DeepHMI performs better than the commonly used bicubic interpolation method in terms of four evaluation metrics. In addition, we demonstrate the ability of DeepHMI through a case study of the enhancement of SDO/HMI transverse and line-of-sight magnetograms of AR 12371 to GST/NIRIS data.


## Project Structure

- `model.py`: Model architecture.
- `train.py`: Training script for the model.
- `test.py`: Testing script to evaluate the model.
- `dataset_train`: Directory containing training data.
- `dataset_test`: Directory containing testing data.
- `model_out`: Directory containing trained model.
- `sr_out`: Directory containing super-resolution results.
- `requirements.txt`: Python dependencies.
- `prediction_workflow.ipynb`: Workflow of prediction.
- `README.md`: Documentation for setup and usage.


## Requirements
Dependencies are listed in `requirements.txt`:

```txt
pytorch==1.12.1
torchvision==0.13.1
astropy<5.0
einops==0.7.0
```

You can install them with:
```bash
conda create -n deephmi python=3.8
conda activate deephmi
pip install -r requirements.txt
```


## Training

Run training:
```bash
python train.py
```

Trained models are saved under:
```
model_out/
```

## Testing / Inference

Run testing:
```bash
python test.py
```

Outputs are saved under:
```
sr_out/
```

Pretrained models are saved under:
```
model_out/
```

The code of loading pretrained models is in test.py file.

## Reference

- Xu, C., Xu, Y., Wang, J. T. L., Li, Q., & Wang, H. (2025). Improving the spatial resolution of SDO/HMI transverse and line-of-sight magnetograms using GST/NIRIS data with machine learning. Astronomy & Astrophysics. https://doi.org/10.1051/0004-6361/202453581 [https://www.aanda.org/articles/aa/full_html/2025/05/aa53581-24/aa53581-24.html](https://www.aanda.org/articles/aa/full_html/2025/05/aa53581-24/aa53581-24.html)


