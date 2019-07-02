import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        """
        Parameters
        - embed_size : Size of Embedding
        - hidden_size : Number of hidden layer nodes 
        - vocab_size : size of vocabrary size
        - num_layers : num of layer
        """
        super(DecoderRNN, self).__init__()

        # Initializing hidden_size 
        self.hidden_size = hidden_size

        # Initializing vocab_size
        self.vocab_size = vocab_size

        # Initializing embedding layer
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        
        # Initializing LSTM layer
        self.lstm = nn.LSTM(input_size = embed_size, 
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            batch_first = True)

        # Initializing output layer                    
        self.linear = nn.Linear(in_features = hidden_size,
                               out_features = vocab_size)
    
    def forward(self, features, captions):
        """
        Parameter
        - features : feature of Images [Batch size, 300]
        - captions : captions [Batch size, 12~19]
        """
        captions = captions[:, :-1] # remove token [1, 1, ... , 1]
        embedding = self.embed(captions) # [batch_size, sentense_len] -> [batch_size, sentense_len, embed_size]
        embedding = torch.cat((features.unsqueeze(1), embedding), dim=1) # [batch_size, sentense_len, embed_size] -> [batch_size, 1+sentense_len, embed_size]
        lstm_out, hidden = self.lstm(embedding) # [batch_size, 1+sentense_len, embed_size] -> [batch_size, 1+sentense_len, hidden_size]
        output = self.linear(lstm_out) # [batch_size, 1+sentense_len, hidden_size] -> # [batch_size, 1+sentense_len, vocab_size] 
        return output

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        predicted_sentence = []

        for index in range(max_len):
            lstm_out, states = self.lstm(input, states)

            lstm_out = lstm_out.squeeze(1)
            outputs = self.liner(lstm_out)

            target = outputs.max(1)[1]

            predicted_sentence.append(target.item())

            inputs = self.embed(target).unsqueeze(1)
