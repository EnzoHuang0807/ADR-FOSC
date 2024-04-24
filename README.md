# Annealing Self-Distillation Rectification (Modified)

## Requirements

The code has been implemented and tested with `Python 3.9.14`
To install requirements:

```setup
pip install -r requirements.txt
```

## Directory Layout

	.
	|__ src # Source files
	|__ data # Directory to put data
	|     |__ cifar10
	|     |__ cifar100
	|     |__ tiny-imagenet-200
	|
	|__ config # Directory to store experiment configs
	|__ *_experiment # Directory to store experiment checkpoints

## Data
Before start training, manully create `data/` directory and downloaded the required data to `cifar10/`, `cifar100/` ([cifar10 and cifar100](https://www.cs.toronto.edu/~kriz/cifar.html)), and `tiny-imagenet-200` directory ([tiny-imagenet-200](https://gist.github.com/moskomule/2e6a9a463f50447beca4e64ab4699ac4)).

If you wish to use additional DDPM sysnthetic data for experiment, please refer to the original paper [Rebuffi et al., 2021](https://arxiv.org/abs/2103.01946). The synthetic data is public available [here](https://github.com/deepmind/deepmind-research/tree/master/adversarial_robustness).

## Training

To train the model(s) in the paper, run this command:

```train
python train.py
	--description <experiment name and other description>
	--gin_config <absolute path to experiment configs>
	--cuda <cuda id>
	--num_workers <how many workers used in data loader>
	--batch_size <batch size>
	--aux_batch_size <synthetic data batch size, specify when using addtional synthetic data>
	--ema <boolean optional, evaluate on the ema teacher is specified>
```

The parameters used in the original paper can be found in the `config/` directory

## Evaluation

To do evaluation on the trained model, run:

```eval
python robust_eval.py 
	--dataset <the dataset used to evaluate>
	--model_type <model type for the checkpoint, should be one of resnet18, preact-resnet18, wideresnet-34-10>
	--model_path <path to the checkpoint>
	--activation_name <activation for the checkpoint, should be one of relu or swish>
	--attack_type <type of attack to evaluate, should be one of fgsm, pgd, autoattack, square>
	--epsilon <epsilon budget (in 255 range) used for evaluation, 8 as default setting>
	--steps <number of steps to attack for pgd evaluation. This argument is not needed for other attacks.>
	--cuda <cuda id>
	--batch_size <batch size>
	--ema <boolean optional, evaluate on the ema teacher is specified>
```
