# Swin Trandormer with Feature Pyramid

## Description

This folder contains the following models

- naive swin transformer model  `models.swintransformer.SwinTransformerHeatmap`
- final swin transformer model `models.swintransformer.SwinTransformerPyramid`

and code used for training those models and generating predictions.

## Model Details

- summing features to save memory
- using layernorm to normalize along each dimension

## Required Package

- `conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`
- `conda install timm pandas matplotlib`
- `conda install -c conda-forge opencv`

