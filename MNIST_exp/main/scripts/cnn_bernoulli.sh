# for ALPHA in 0.72 0.76 0.80 0.84 0.88 0.92 0.96
for ALPHA in 0.80
do
    echo "python code/train.py mnist mnist_cnn models/mnist/cnn/alpha_${ALPHA} --batch 400 --perturb bernoulli --alpha ${ALPHA} --epochs 30 --lr_step_size 10 --print-freq 200 --optimizer nesterov --tune --lr 0.05"
    python code/train.py mnist mnist_cnn models/mnist/cnn/alpha_${ALPHA} --batch 400 --perturb bernoulli --alpha ${ALPHA} --epochs 30 --lr_step_size 10 --print-freq 200 --optimizer nesterov --tune --lr 0.05
    echo ""
done

N=100000
BZ=50000
DIR=models/mnist/cnn/alpha_
# for ALPHA in 0.72 0.76 0.80 0.84 0.88 0.92 0.96
for ALPHA in 0.80
do
    # python code/discrete_certify.py mnist ${DIR}${ALPHA}/checkpoint.best.tar ${ALPHA} ${DIR}${ALPHA}/valid.txt --skip 1 --batch ${BZ} --N ${N} --N0 100 --split valid
    # echo ""
    
    python code/discrete_certify.py mnist ${DIR}${ALPHA}/checkpoint.best.tar ${ALPHA} ${DIR}${ALPHA}/test.txt --skip 1 --batch ${BZ} --N ${N} --N0 100 --split test
    echo ""
done

