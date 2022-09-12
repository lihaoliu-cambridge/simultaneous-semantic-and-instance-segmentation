# Simultaneous Semantic and Instance Segmentation for Colon Nuclei Identification and Counting

by Lihao Liu, Chenyang Hong, Angelica I. Aviles-Rivero and Carola-Bibiane Schonlieb.  


## Introduction

In this repository, we provide Pytorch implementation for the 4th solution in the grand challenge [CoNIC-2022](https://conic-challenge.grand-challenge.org/). The detailed description can be found in our MIUA-22 paper [Simultaneous Semantic and Instance Segmentation for Colon Nuclei Identification and Counting](https://arxiv.org/pdf/2203.00157.pdf). 

<img src="https://github.com/lihaoliu-cambridge/lihaoliu-cambridge.github.io/blob/master/pic/papers/ssis_visual_results.png">  


## Requirement

pytorch.             1.10.1  
cuda                 11.0
cudnn                8.0

To install detectron2, you need to install the it from my repo, since I make changes to the orginl detectron2:
   ```shell
   cd detectron2
   python setup.py install
   ```  


## Code Structure

There are three parts (three subdirs) in this repo, which are exactly the same three parts listed in the main paper: 

1. The instance model in subdir [detectron2](https://github.com/lihaoliu-cambridge/simultaneous_semantic_and_instance_segmentation/tree/main/detectron2), which contrains the training and testing logic of the Cascade Mask-RCNN 152 model forked and modified from the original detectron2.
   
2. The semantic model in subdir [hover_net](https://github.com/lihaoliu-cambridge/simultaneous_semantic_and_instance_segmentation/tree/main/hover_net), which contrains the training and testing logic of the Hover-Net model.
   
3. The subdir [conic](https://github.com/lihaoliu-cambridge/simultaneous_semantic_and_instance_segmentation/tree/main/conic), which contrains the data pre-processing and NMS embedding modole.

<img src="https://github.com/lihaoliu-cambridge/lihaoliu-cambridge.github.io/blob/master/pic/papers/ssis_network.png">  


## Usage

1. Download the dataset from the challenge website: https://conic-challenge.grand-challenge.org/, and unzip everything in folder `./conic/dataset`.

2. Run the three pre-processing files in `./conic/utils/` in order,
   
   ```shell
   cd script
   python 1_generate_5fold_split.py
   python 2_turn_npy_to_png.py
   python 3_generate_coco_format_data.py
   ```  
   The generated input files for detectron2 and hovernet are stored in `./conic/dataset/detectron2` and `./conic/dataset/hovernet`, respectively.

3. Run the instance model in subdir [detectron2](https://github.com/lihaoliu-cambridge/simultaneous_semantic_and_instance_segmentation/tree/main/detectron2):

   ```shell
   cd detectron2
   ./tools/train_net.py --num-gpus 4 --config-file ./configs/Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml
   ```  
   Note: you need to change the batch_size, training steps parameters accordingly in  line 29-36 in `./detectron2/configs/Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml` before training. I used 4 A100 GPUs with 80 GB memory. My batch size is 96 and max step is 3200. If you decrease the batch size, please increase the max iteration, checkpoint period, and so on. For instance, 
   
   ```shell
     IMS_PER_BATCH: 8        # 8 = 96 / 12
     STEPS: (28800, 22600)   # (1800 * 12, 2200 * 12)
     MAX_ITER: 38400         # 3200 * 12
     BASE_LR: 0.04
     WARMUP_ITERS: 2400      # 200 * 12
     WARMUP_FACTOR: 0.01
     CHECKPOINT_PERIOD: 2400 # 200 * 12
   ```  
   
4. Run the semantic model in subdir [hover_net](https://github.com/lihaoliu-cambridge/simultaneous_semantic_and_instance_segmentation/tree/main/hover_net):

   ```shell
   cd hovernet
   python run.py --gpu '0,1,2,3' --fold 0
   ```  
   Note: you need to change the batch_size, nr_procs, and pretrained_backbon parameters in `./hover_net/param/template.yaml` before training
   
   
5. Run the NMS code after on the predicted results on 3 & 4:

   ```shell
   cd conic
   python ensemble_semantic_and_instance_prediction_with_nms.py --instance_pred your_instance_test_result.npy --semantic_pred your_instance_test_result.npy --out_dir_path your_output_dir --fold 0
   ``` 
   
   
## Citation

If you use our code for your research, please cite our paper:

```
@article{liu2022simultaneous,
  title={Simultaneous Semantic and Instance Segmentation for Colon Nuclei Identification and Counting},
  author={Liu, Lihao and Hong, Chenyang and Aviles-Rivero, Angelica I and Sch{\"o}nlieb, Carola-Bibiane},
  journal={arXiv preprint arXiv:2203.00157},
  year={2022}
}
```


## Question

Please open an issue or email lhliu1994@gmail.com for any questions.
