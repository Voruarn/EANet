# EANet, IEEE TGRS 2025</a> </p>

- Paper: [Rethinking masked autoencoder for salient object detection in optical remote sensing images from a pseudo image pretraining perspective](https://ieeexplore.ieee.org/document/10981795)


## Abstract

Recently masked autoencoder (MAE) has achieved great success in visual representation learning and delivered promising potential in many downstream vision tasks. However, due to the lack of a saliency supervision signal in the original MAE, almost no saliency information can be learned from the masked image reconstruction process. Therefore, salient object detection (SOD) can hardly benefit from MAE pretraining. To address this issue, we integrate the SOD model and saliency supervision in MAE and propose a simple and effective framework for dynamic hybrid masking MAE (DHMMAE), utilizing MAE with dynamic hybrid masking (DHM) ratios to pretrain SOD model. Specifically, we treat MAE as an online data augmenter to generate endless pseudo images for the SOD model to predict saliency maps, where saliency supervision is employed to ensure that the model can learn robust saliency prior knowledge from the reconstructed images. Besides, we propose a simple and novel network EANet driven by DHMMAE pretraining for SOD in optical remote sensing images (ORSIs). Two key modules are designed for EANet to further enhance its performance: the enhanced diverse feature aggregation module (EDFAM) and the adjacent-context shuffle spatial attention module (ASSAM). EDFAM aggregates diverse features via three different types of convolution layers and enhances them by convolution block attention module (CBAM). ASSAM captures spatial location information of salient objects by employing channel shuffle operation and weighted spatial attention mechanism on the fused adjacent context. Experiments on three ORSI-SOD datasets demonstrate that our proposed method outperforms the cutting-edge methods. Code is available at https://github.com/Voruarn/EANet.


```
## 📎 Citation

If you find the code helpful in your research or work, please cite the following paper(s).

@ARTICLE{10981795,
  author={Fu, Yuxiang and Fang, Wei and Sheng, Victor S.},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Rethinking Masked Autoencoder for Salient Object Detection in Optical Remote Sensing Images From a Pseudo Image Pretraining Perspective}, 
  year={2025},
  volume={63},
  number={},
  pages={1-12},
 }

```
