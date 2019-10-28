N=100000
#N=1000
BZ=1000
DIR=models/imagenet/resnet50/alpha_
ALPHA=0.2
python code/discrete_certify.py imagenet ${DIR}${ALPHA}/checkpoint.pth.tar ${ALPHA} ${DIR}${ALPHA}/test.txt --skip 100 --batch ${BZ} --N ${N} --N0 100 --split test 



