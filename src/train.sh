python train.py \
	--description TRADES-AT-wrn-random-0.2 \
	--show_plot \
	--random_ratio 0.2 \
	--cuda 6 \
	--gin_config ./config/cifar10/wrn_trades_sgd.gin

python train.py \
	--description preact-PGD-AT-AWP-ADR \
	--cuda 2 \
	--gin_config ./config/cifar10/preact-resnet18_pgd_awp_adr.gin


python train.py \
	--description TRADES-AT-ADR-5e-3 \
	--cuda 2 \
	--gin_config ./config/cifar10/resnet18_trades_sgd_adr.gin
