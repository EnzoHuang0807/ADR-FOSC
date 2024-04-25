python train.py \
	--description PGD-AT \
	--cuda 2 \
	--gin_config ./config/cifar10/resnet18_pgd_sgd.gin

python train.py \
	--description PGD-AT-ADR \
	--cuda 5 \
	--gin_config ./config/cifar10/resnet18_pgd_sgd_adr.gin

python train.py \
	--description PGD-AT-AWP \
	--cuda 6 \
	--gin_config ./config/cifar10/resnet18_pgd_awp.gin

python train.py \
	--description PGD-AT-AWP-ADR \
	--cuda 7 \
	--gin_config ./config/cifar10/resnet18_pgd_awp_adr.gin
