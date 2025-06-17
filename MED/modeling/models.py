import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    
    """
    A simple feature extractor (MLP) that applies a linear transformation to the input data.
    This is used to extract features from the image data before passing it to the CNN.
    """

    def __init__(self, 
                 input_dim: int = 2048,
                 output_dim: int = 32,
                 hidden_dims: list = None):
    
        super(FeatureExtractor, self).__init__()

        self.linear = nn.Sequential()
        for i in range(len(hidden_dims)):
            if i == 0:
                self.linear.add_module(f'linear_{i}', nn.Linear(input_dim, hidden_dims[i]))
            
            else:
                self.linear.add_module(f'linear_{i}', nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        
        self.linear.add_module('output', nn.Linear(hidden_dims[-1], output_dim))
    
    def forward(self, x):
        return self.linear(x)
    

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                
            try: 
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)
            except:
                pass

class CNN(nn.Module):
    """
    A Convolutional Neural Network (CNN) for multimodal (image + kinematic) data processing.
    This model consists of two convolutional layers followed by fully connected layers.
    """

    def __init__(self, 
                 in_features: int = 58, # 26 kinematic + 32 image features
                 window_size:int = 30):

        super(CNN,self).__init__()
        
        self.name = "SimpleCNN"
        self.window_size = window_size  
        if self.window_size == 10: #10 sample window limits convolutional layer size
            self.convolutional_layers = nn.Sequential(
                nn.Conv1d(in_features, 64, kernel_size=3,stride=1),
                nn.MaxPool1d(2,2),
                nn.Dropout(p=0.2),
                nn.BatchNorm1d(64),
                nn.Conv1d(64, 128,kernel_size=3,stride=1),
                nn.MaxPool1d(2,2),
                nn.Dropout(p=0.2),
                nn.BatchNorm1d(128),
                nn.Flatten())
        
        elif self.window_size == 30: #30 sample window allows more depth in convolutional layers
            self.convolutional_layers = nn.Sequential(
                nn.Conv1d(26 + 32, 64, kernel_size=3,stride=1),
                nn.MaxPool1d(2,2),
                nn.Dropout(p=0.2),
                nn.BatchNorm1d(64),
                nn.Conv1d(64, 128,kernel_size=3,stride=1),
                nn.MaxPool1d(2,2),
                nn.Dropout(p=0.2),
                nn.BatchNorm1d(128),
                nn.Conv1d(128, 256,kernel_size=3,stride=1),
                nn.MaxPool1d(2,2),
                nn.Dropout(p=0.2),
                nn.BatchNorm1d(256),
                nn.Flatten())
        
        
        #Compute output size
        with torch.no_grad():
            self.convolutional_layers.eval()
            dummy_input = torch.zeros(1, 58, self.window_size)
            n_features = self.convolutional_layers(dummy_input).shape[1]

        
        self.linear_layers = nn.Sequential(nn.Linear(n_features,256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256,32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16,1))

        self.initialize_weights()
        
    
    def forward(self, x):

        x = self.convolutional_layers(x)
        x = self.linear_layers(x)

        return x
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.1)



class LSTM(nn.Module):

    """
    A Long Short-Term Memory (LSTM) network for processing sequential data.
    This model consists of three LSTM layers followed by fully connected layers.
    """
    
    def __init__(self,
                 in_features: int = 58,
                 window_size: int = 30):
        
        super(LSTM, self).__init__()
        self.name = "SimpleLSTM"
        self.window_size = window_size
        self.in_features = in_features
       
        self.layer_dim =1
        self.lstm1 = nn.LSTM(self.in_features, 512, dropout=0, num_layers=self.layer_dim,batch_first=True)
        self.lstm2 = nn.LSTM(512, 128, dropout=0, num_layers=self.layer_dim,batch_first=True)
        self.lstm3 = nn.LSTM(128, 64, dropout=0, num_layers=self.layer_dim,batch_first=True)

        #Compute output size
        with torch.no_grad():
            self.lstm1.eval()
            dummy_input = torch.zeros(1, self.window_size, self.in_features)
            lstm_out, _ = self.lstm1(dummy_input)
            lstm_out, _ = self.lstm2(lstm_out)
            lstm_out, _ = self.lstm3(lstm_out)
            n_features = lstm_out.shape[2] * lstm_out.shape[1]

        self.linear_layers = nn.Sequential(nn.Flatten(),
                                           nn.Linear(n_features,960),
                                           nn.ReLU(),
                                           nn.Dropout(p=0.55),
                                           nn.Linear(960,480),
                                           nn.ReLU(),
                                           nn.Dropout(p=0.55),
                                           nn.Linear(480,16),
                                           nn.ReLU(),
                                           nn.Linear(16,1))
        
        self.initialize_weights()

    def forward(self,l):
            
        #LSTM 1
        l = l.transpose(1, 2).contiguous()
        lstm, _ = self.lstm1(l)
        lstm = F.relu(lstm)
        
        #LSTM 2
        lstm, _ = self.lstm2(lstm)
        lstm = F.relu(lstm)
        
        #LSTM 3
        lstm,_ = self.lstm3(lstm)
        lstm = F.relu(lstm)

        #Flatten and apply linear layers
        lstm = self.linear_layers(lstm) 
        return lstm
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias,0)


class Siamese_CNN(nn.Module):
    """
    A Siamese Convolutional Neural Network (CNN) for processing image data.
    This model consists in:
    1. Two identical CNN convolutional branches that share weights.
    2. The output from each branch is subtracted (and absolute value) to compute the similarity between the two inputs.
    3. A final linear layer is applied to this output to produce the final prediction.

    """

    def __init__(self,
                 in_features: int = 58,
                  window_size: int = 30):
        super(Siamese_CNN, self).__init__()
        self.name = "Siamese_CNN"
        self.cnn_branch = CNN(window_size=window_size, in_features=in_features)
        self.convolutional_layers = self.cnn_branch.convolutional_layers
        self.linear_layers = self.cnn_branch.linear_layers

    def forward(self, x1, x2):
        out1 = self.convolutional_layers(x1)   
        out2 = self.convolutional_layers(x2)
        
        diff = torch.abs(out1 - out2)
        output = self.linear_layers(diff)

        return output
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    

class Siamese_LSTM(nn.Module):
    """
    A Siamese Long Short-Term Memory (LSTM) network for processing sequential data.
    This model consists in:
    1. Two identical LSTM branches that share weights.
    2. The output from each branch is subtracted (and absolute value) to compute the similarity between the two inputs.
    3. A final linear layer is applied to this output to produce the final prediction.
    
    """

    def __init__(self, in_features: int = 58, window_size: int = 30):
        super(Siamese_LSTM, self).__init__()
        self.name = "Siamese_LSTM"

        self.lstm = LSTM(in_features=in_features, window_size=window_size)
        self.lstm1 = self.lstm.lstm1
        self.lstm2 = self.lstm.lstm2
        self.lstm3 = self.lstm.lstm3
        self.linear_layers = self.lstm.linear_layers

    def forward(self, x1, x2):
        #LSTM 1
        x1 = x1.transpose(1, 2).contiguous()
        x2 = x2.transpose(1, 2).contiguous()
        lstm_x1, _ = self.lstm1(x1)
        lstm_x1 = F.relu(lstm_x1)
        lstm_x2, _ = self.lstm1(x2)
        lstm_x2 = F.relu(lstm_x2)
        
        #LSTM 2
        lstm_x1, _ = self.lstm2(lstm_x1)
        lstm_x1 = F.relu(lstm_x1)
        lstm_x2, _ = self.lstm2(lstm_x2)
        lstm_x2 = F.relu(lstm_x2)
        
        #LSTM 3
        lstm_x1, _ = self.lstm3(lstm_x1)
        lstm_x1 = F.relu(lstm_x1)
        lstm_x2, _ = self.lstm3(lstm_x2)
        lstm_x2 = F.relu(lstm_x2)

        #Flatten and apply linear layers
        out = torch.abs(lstm_x1 - lstm_x2)
        output = self.linear_layers(out)

        return output
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)