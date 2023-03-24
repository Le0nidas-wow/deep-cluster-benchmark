# deep-cluster-benchmark
eazy deep cluster algorithm (DEKM/IDEC/DCN)
## Requirements 
+ pytorch
+ torchversion
+ sklearn
+ numpy
+ argparse
## Usage
open your cmd and input:  
git clone https://github.com/Le0nidas-wow/deep-cluster-benchmark.git  
cd ./deep-cluster-benchmark  
python main.py  
if you want to change the parameter  
you can   
  
python main.py --model DEKM --dataset MNIST --lr 0.01 --momentum 0.9 --weight_decay 5e-4 --epoch 20  
  
you can choose MNIST/USPS/SVHN as your dataset  
you can choose DEKM/IDEC/DCN as your model  
  
after you press your ENTER  
you can see the loss and acc of each epoch and the acc,ari and nmi of the test dataset.  
  
more dataset will release in the future.
oops, the benchmark function has not implemented yet.
## Thanks
*Chatgpt*  
*nvidia cuda*  
*pytorch*  
*python*  
and other developers  




