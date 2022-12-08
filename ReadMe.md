Necessary libraries:
1. Install conda using 
    pip install conda
    
2. We provide a environment.yml file that contains the packages required. Running the following command will duplicate the conda environment under the name myenv .
    conda env create -f environment.yml

3. Activate the conda environment using
   conda activate myenv

4. Set up an account in weights and biases for cloud based logging. Install wandb using 
    pip install wandb
   Login using
    wandb login





######
1. To train a model on CIFAR-10, run the following command (after replacing (a) $model with either resnet56 or vgg16_bn, (b) $momentum with the momentum parameter, (c) $rho with the parameter for the running average in the denominator, (d) $lr with the desired learning rate, (e) $epsilon with the epsilon parameter for adam, (f) $batch_size with the desired batch size, (g) $save_dir with the directory to save checkpoints, (h) $wandb_project with weights and biases project to add your results to, (i) $wandb_entity with weights and biases entity to add your results to, (j) $schedule_pattern with the lr scheduling pattern ('300-400-500' for rmsprop, '300-500' for adam), (k) $algo with one of rmsprop or adam, (l) $data_path with the correct path to cifar-10 files ):

python -u cifar_main.py --data_dir $data_path --arch=$model  --save-dir=$save_dir --weight-decay=1e-4 --loss_type='xent' --momentum=$momentum --epochs=500 --lr=$lr --rho=$rho --optimizer=$algo --batch-size=$batch_size --schedule_lr=1 --warmup=1 --warmup_steps=10 --schedule_pattern=$schedule_pattern --looper=1 --epsilon=$epsilon --train_final_layer=0 --wandb_project $wandb_project --wandb_entity $wandb_entity


######
2. To run the SVAG experiments on CIFAR-10, run the following command (after replacing (a) $model with either resnet56 or vgg16_bn, (b) $momentum with the momentum parameter, (c) $rho with the parameter for the running average in the denominator, (d) $lr with the desired learning rate, (e) $epsilon with the epsilon parameter for adam, (f) $batch_size with the desired batch size, (g) $save_dir with the directory to save checkpoints, (h) $wandb_project with weights and biases project to add your results to, (i) $wandb_entity with weights and biases entity to add your results to, (j) $schedule_pattern with the lr scheduling pattern ('300-400-500' for rmsprop, '300-500' for adam), (k) $algo with one of rmsprop or adam, (l) $svag_param with the svag parameter (1, 2, 4, 8 in the paper), (m) $data_path with the correct path to cifar-10 files ):


python -u cifar_main.py --data_dir $data_path --arch=$model --svag_param=$svag_param  --save-dir=$save_dir  --weight-decay=1e-4 --loss_type='xent' --momentum=$momentem --epochs=500 --lr=$lr --rho=$rho --optimizer=$algo --batch-size=$batch_size --schedule_lr=1 --warmup=1 --warmup_steps=10 --schedule_pattern=$schedule_pattern --looper=1 --epsilon=$epsilon --train_final_layer=0  --wandb_project $wandb_project --wandb_entity $wandb_entity



######
3. To run Imagenet experiments, run the following command (after replacing (a) $data_path with the correct path to imagenet files, (b) $momentum with the momentum parameter, (c) $rho with the parameter for the running average in the denominator, (d) $lr with the desired learning rate, (e) $epsilon with the epsilon parameter for adam, (f) $batch_size with the desired batch size, (g) $wandb_project with weights and biases project to add your results to, (h) $wandb_entity with weights and biases entity to add your results to ):

python Imagenet_main.py -b $batch_size -a resnet50 --optimizer adam --momentum $momentum --rho $rho --lr $lr --warmup 1 --epsilon $epsilon --warmup_steps 5 --schedule_lr 1 --schedule_pattern='50-80-90' --gpu=0 --sample_mode random_shuffling --world-size 1 --rank 0 $data_path --workers 10 --wandb_project $wandb_project --wandb_entity $wandb_entity
