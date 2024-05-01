python train.py \
	--description PGD-AT \
	--cuda 1 \
	--gin_config ./config/cifar10/resnet18_pgd_sgd.gin

python train.py \
	--description PGD-AT-ADR \
	--cuda 2 \
	--gin_config ./config/cifar10/resnet18_pgd_sgd_adr.gin

python train.py \
	--description preact-PGD-AT-AWP \
	--cuda 4 \
	--gin_config ./config/cifar10/preact-resnet18_pgd_awp.gin

python train.py \
	--description preact-PGD-AT-AWP-ADR \
	--cuda 2 \
	--gin_config ./config/cifar10/preact-resnet18_pgd_awp_adr.gin
