# MNIST-Classifier
applying Convolutional Neural Network with MNIST data to classify the digits using pytorch library. 
### Accuracy: 97% 
---
```python
class Convolutional_Neural_Network(nn.Module):
    def __init__(self):
        super(Convultional_Neural_Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv_dropout = nn.Dropout2d()
        self.fully_connected_layer = nn.Linear(320, 50)
        self.fully_connected_layer2 = nn.Linear(50, 10)

    def forward(self, x):
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv1(x), 2))
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv_dropout(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = nn.functional.relu(self.fully_connected_layer(x))
        x = nn.functional.dropout(x , training= self.training)
        x = self.fully_connected_layer2(x) 
        return nn.functional.softmax(x)
```
![visualize](https://github.com/Esmail-ibraheem/MNIST-Classifier/assets/113830751/c9a1de97-93e8-437a-8b92-a81666219547)

---
### Download the MNIST data: 
https://github.com/Esmail-ibraheem/MNIST-data

---
### output: 
![image](https://github.com/Esmail-ibraheem/MNIST-Classifier/assets/113830751/5a0efba0-0398-4d35-81c4-35f1ea029767) ![image](https://github.com/Esmail-ibraheem/MNIST-Classifier/assets/113830751/41efb9c2-0352-4b59-892f-c2c1a7e8d741) ![image](https://github.com/Esmail-ibraheem/MNIST-Classifier/assets/113830751/52cb7c55-f18a-4aa4-a367-4b1a5ff661a8) ![image](https://github.com/Esmail-ibraheem/MNIST-Classifier/assets/113830751/15eae0d0-58f3-4216-88f1-bf18dda0fb56) ![image](https://github.com/Esmail-ibraheem/MNIST-Classifier/assets/113830751/78f00bf4-202e-4897-8f30-fe2d6ba38938) ![image](https://github.com/Esmail-ibraheem/MNIST-Classifier/assets/113830751/1ad53868-0f75-4296-ae58-edbf2be0c04f)






