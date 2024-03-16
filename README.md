# Generating Cifar10 dataset images in the desired class using DDPM, DDIM, and Classifier-Free guided
### How to change the dataset from ImageNet to Cifar10 in [the repository](https://github.com/gmongaras/Diffusion_models_from_scratch) in order to produce images with a diffusion model?

[This repo](https://github.com/gmongaras/Diffusion_models_from_scratch)  is composed of DDPM, DDIM, and Classifier-Free guided models trained on ImageNet 64x64.
<br><br>
![#1589F0](https://via.placeholder.com/15/1589F0/1589F0.png)Our target: To produce Cifar10 dataset images using diffusion in the desired class.
<br>You can run the project according to the instructions below.

- 1-	Upload the DiffusionWithTrainCifar10 file in Colab and start running from the beginning.
- 2-	After cloning the specified repository(in line 2), replace the following files with the original files of the repository and continue the execution. In this way, we change the dataset to Cifar10.<br>
<br>loadImagenet64(in the data folder)
make_massive_tensor (in the data folder)
model_trainer (in the src folder)

 - 3-	#res, res, clsAtn, atn, chnAtn: I chose this model based on the recommendation of the used repository, but other network models can also be used.

```
!python src/train.py --blk_types res,res,clsAtn,atn,chnAtn
```
- 4-	After the training is over and the weights are obtained, we produce a image in the desired class to check its accuracy.
```
!python -m src.infer --loadDir models --loadFile model_4e_3128s.pkl --loadDefFile model_params_4e_3128s.json --device gpu --step_size 20 --class_label 1 --out_imgname 'test.png' --out_gifname 'test.gif'
```
- Note: model_4e_3128s.pkl and model_params_4e_3128s.json are created in the models folder after learning is done.

![#1589F0](https://via.placeholder.com/15/1589F0/1589F0.png)Due to our limitations, we performed with epochs=50 and the real photo was not produced. The network will need more Fine-Tuning and training which requires further investigation.
<br>![#1589F0](https://via.placeholder.com/15/1589F0/1589F0.png)numSaveSteps: This parameter determines the number of times the weights are saved. 
<br>![#1589F0](https://via.placeholder.com/15/1589F0/1589F0.png) Each time the weights are saved, about 2.5 GB of space is required. We set this parameter depending on the need and space.
If numSaveSteps=782, it means that weights is saved once in every epoch.
  
