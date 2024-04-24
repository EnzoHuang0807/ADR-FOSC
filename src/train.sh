python train.py \
	--description PGD-AT \
	--cuda 0 \
	--gin_config ./config/cifar10/resnet18_pgd_sgd.gin

python train.py \
	--description PGD-AT-ADR \
	--cuda 0 \
	--gin_config ./config/cifar10/resnet18_pgd_sgd_adr.gin

python train.py \
	--description PGD-AT-AWP \
	--cuda 0 \
	--gin_config ./config/cifar10/resnet18_pgd_awp.gin

python train.py \
	--description PGD-AT-AWP-ADR \
	--cuda 0 \
	--gin_config ./config/cifar10/resnet18_pgd_awp_adr.gin
