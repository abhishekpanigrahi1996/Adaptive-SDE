# SDEs for Adaptive algorithms

This is the implementation for the paper [On the SDEs and Scaling Rules for Adaptive Gradient Algorithms](https://arxiv.org/abs/2205.10287)


## Installation
Please install conda. We provide a environment.yml file that contains the packages required. Running the following command will duplicate the conda environment under the name myenv .
```
    conda env create -f environment.yml
```
We use weights and biases to keep track of model behavior along the trajectory.




## Run the code

1. To train a model on CIFAR-10, run the following command 
```
python -u cifar_main.py  \
              --data_dir $data_path  \
              --arch=$model  \
              --save-dir=$save_dir \
              --weight-decay=1e-4 \
              --loss_type='xent' \
              --momentum=$momentum \
              --epochs=500 \
              --lr=$lr \
              --rho=$rho \
              --optimizer=$algo \
              --batch-size=$batch_size \
              --schedule_lr=1 \
              --warmup=1 \
              --warmup_steps=10 \
              --schedule_pattern=$schedule_pattern \
              --looper=1 \
              --epsilon=$epsilon \
              --train_final_layer=0 \
              --wandb_project $wandb_project \
              --wandb_entity $wandb_entity
```
~~~
1.  model: the model to run  ( resnet56 / vgg16_bn )
2.  momentum:  the momentum parameter of adaptive algorithms  
3.  rho:  parameter of the running average of the second order moments for adaptive algorithms 
4.  lr: Learning rate
5.  schedule_pattern: Learning rate schedule pattern ('300-400-500' for rmsprop, '300-500' for adam)
6.  algo: Algorithm to run (rmsprop / adam)
7.  epsilon: Epsilon parameter for adaptive algorithms 
8.  batch_size: Batch size for training
9.  save_dir: Directory to save checkpoints
10. wandb_project: weights and biases project to store the progress
11. wandb_entity:   Project file inside the  weights and biases project
12. data_path: Path to Cifar-10 files
~~~



2. To run the SVAG experiments on CIFAR-10, run the following command 

```
python -u cifar_main.py \
		--data_dir $data_path \
		--arch=$model \
		--svag_param=$svag_param \
		--save-dir=$save_dir \
		--weight-decay=1e-4 \
		--loss_type='xent' \
		--momentum=$momentem \
		--epochs=500 \
		--lr=$lr \
		--rho=$rho \
		--optimizer=$algo \
		--batch-size=$batch_size \
		--schedule_lr=1 \
		--warmup=1\
		--warmup_steps=10 \
		--schedule_pattern=$schedule_pattern \
		--looper=1 \
		--epsilon=$epsilon \
		--train_final_layer=0  \
		--wandb_project $wandb_project \
		--wandb_entity $wandb_entity
```
~~~
1.  model: the model to run  ( resnet56 / vgg16_bn )
2.  momentum:  the momentum parameter of adaptive algorithms  
3.  rho:  parameter of the running average of the second order moments for adaptive algorithms 
4.  lr: Learning rate
5.  schedule_pattern: Learning rate schedule pattern ('300-400-500' for rmsprop, '300-500' for adam)
6.  algo: Algorithm to run (rmsprop / adam)
7.  epsilon: Epsilon parameter for adaptive algorithms 
8.  batch_size: Batch size for training
9.  save_dir: Directory to save checkpoints
10. wandb_project: weights and biases project to store the progress
11. wandb_entity:   Project file inside the  weights and biases project
12. data_path: Path to Cifar-10 files
13. svag_param: SVAG parameters (1, 2, 4, 8 in the paper)
~~~






######
3. To run Imagenet experiments, run the following command 
```
python Imagenet_main.py\
		 -b $batch_size\
		 -a resnet50 \
		 --optimizer adam \
		 --momentum $momentum \
		 --rho $rho \
		 --lr $lr \
		 --warmup 1 \
		 --epsilon $epsilon \
		 --warmup_steps 5 \
		 --schedule_lr 1 \
		 --schedule_pattern='50-80-90' \
		 --gpu=0 \
		 --sample_mode random_shuffling \
		 --world-size 1 \
		 --rank 0 \
		 $data_path \
		 --workers 10 \
		 --wandb_project $wandb_project \
		 --wandb_entity $wandb_entity
```		 
~~~
1.  momentum:  the momentum parameter of adaptive algorithms  
2.  rho:  parameter of the running average of the second order moments for adaptive algorithms 
3.  lr: Learning rate
4.  epsilon: Epsilon parameter for adaptive algorithms 
5.  batch_size: Batch size for training
6. wandb_project: weights and biases project to store the progress
7. wandb_entity:   Project file inside the  weights and biases project
8. data_path: Path to Imagenet files
~~~








Please cite our work if you make use of our code:

```bibtex
@inproceedings{
    malladi2022on,
    title={On the {SDE}s and Scaling Rules for Adaptive Gradient Algorithms},
    author={Sadhika Malladi and Kaifeng Lyu and Abhishek Panigrahi and Sanjeev Arora},
    booktitle={Advances in Neural Information Processing Systems},
    editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
    year={2022},
    url={https://openreview.net/forum?id=F2mhzjHkQP}
}
```

