# FRRNet
# Training/Testing
The training and testing experiments are conducted using PyTorch with a single GeForce NVIDIA GTX 4090 Ti of 6 GB Memory.
1. Prerequisites:
Note that BBNet is only tested on Ubuntu OS with the following environments. It may work on other operating systems (i.e., Windows) as well but we do not guarantee that it will.
Installing necessary packages: PyTorch > 1.1, opencv-python
2. Prepare the data:
downloading testing dataset and moving it into ./Dataset/TestDataset/.
downloading training/validation dataset and move it into ./Dataset/TrainDataset/.
CoCOD8K can be download from [here](https://pan.quark.cn/s/5bdc87f4e0c0#/list/share) -->  Code: tdYx
3. Training Configuration:
Assigning your costumed path, like --train_save and --train_path in train.py.
Just enjoy it via run python train.py in your terminal.
4. Testing Configuration:
After you download all the pre-trained models and testing datasets, just run test.py to generate the final prediction map: replace your trained model directory (--pth).
Just enjoy it!
5. Evaluating your trained model:
You can evaluate the result maps using the tool from [here](https://pan.quark.cn/s/e5f0148f77f5#/list/share).








