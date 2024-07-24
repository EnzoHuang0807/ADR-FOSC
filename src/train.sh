python train.py \
	--description PGD-AT-ADR-I \
	--cuda 5 \
	--gin_config ./config/cifar10/resnet18_pgd_sgd_adr.gin

python train.py \
	--description PGD-AT-AWP-ADR \
	--cuda 2 \
	--fosc_threshold 0 \
	--gin_config ./config/cifar10/resnet18_pgd_awp_adr.gin

python train.py \
	--description preact-PGD-AT-AWP-ADR \
	--cuda 2 \
	--gin_config ./config/cifar10/preact-resnet18_pgd_awp_adr.gin

# python train.py \
# 	--description TRADES-AT-AWP \
# 	--cuda 3 \
# 	--gin_config ./config/cifar10/resnet18_trades_awp.gin

# python train.py \
# 	--description TRADES-AT-AWP-ADR \
# 	--cuda 4 \
# 	--gin_config ./config/cifar10/resnet18_trades_awp_adr.gin


python train.py \
	--description TRADES-AT-ADR-2e-2 \
	--cuda 3 \
	--gin_config ./config/cifar10/resnet18_trades_sgd_adr.gin

python train.py \
	--description TRADES-AT-ADR-1e-3 \
	--cuda 1 \
	--gin_config ./config/cifar10/resnet18_trades_sgd_adr.gin

python train.py \
	--description TRADES-AT-ADR-5e-3 \
	--cuda 2 \
	--gin_config ./config/cifar10/resnet18_trades_sgd_adr.gin
