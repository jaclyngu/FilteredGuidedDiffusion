# Filtered-Guided Diffusion for Controllable Image Generation
*Fast, lightweight, architecture-independent, low-level frequency control for diffusion-based Image-to-Image translations.*

[**Filtered-Guided Diffusion for Controllable Image Generation**](http://filterguideddiffusion.github.io/)<br/>
[Zeqi Gu*](https://github.com/jaclyngu),
[Ethan Yang*](https://www.cs.cornell.edu/abe/group/members),
[Abe Davis](http://abedavis.com/)<br/>
<em>\* denotes equal Contribution</em><br/>
_[GitHub](https://github.com/jaclyngu/FilteredGuidedDiffusion) | [Paper](https://dl.acm.org/doi/10.1145/3641519.3657489) | [Project Page](http://filterguideddiffusion.github.io)_


![Teaser](./figures/teaser_updated.png)

<!-- ## Video
#### [Video: <em>(a bit outdated but still relevant. This will be replaced with an updated version.)</em>](https://youtu.be/JQXnEO1aI4I)
[![](./figures/FGDThumbnailYTLarge.jpg)](https://youtu.be/JQXnEO1aI4I) -->

## Code (now released!)
### Summary
We provide a lightweight implementation of FGD which contains all the core functionality described in our paper. Our code is based on the [🤗 diffusers library](https://huggingface.co/docs/diffusers/en/index) and [taichi lang](https://www.taichi-lang.org/) for efficient computation of the cross bilateral matrix. 

A full explanation of how to use our code is described in the jupyter notebook <em>demo.ipynb</em>. 

For questions regarding the code, or access to a more comprehensive set of fuctions (<em>although much less user friendly</em>) including experimental features, debugging, and evaluation, please contact both authors at zg45@cornell.edu and eey8@cornell.edu. 

### Setup 
We provide a <em>requirements.txt</em> file which contains the packages our implementation of FGD was tested on. Note running our code requires a GPU. 
```
diffusers==0.30.0
numpy==2.0.1
Pillow==10.4.0
pytorch_lightning==2.4.0
taichi==1.7.1
torch==2.4.0+cu118
tqdm==4.66.5
transformers==4.44.0
```
**Note:** <em>jupyter notebook is also required in order to run our demo as we do not provide a command line interface.</em>

## Citation
For those wishing to use our work, please use the following citation:
```
@inproceedings{gu2024filter,
  title={Filter-Guided Diffusion for Controllable Image Generation},
  author={Gu, Zeqi and Yang, Ethan and Davis, Abe},
  booktitle={ACM SIGGRAPH 2024 Conference Papers},
  pages={1--10},
  year={2024}
}
```