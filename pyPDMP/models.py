import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(torch.nn.Module):

    def __init__(self,
                 input_dimension,
                 output_dimension,
                 num_neurons=[128, 128, 128],
                 input_normalization=0,
                 hidden_activation='relu',
                 out_activation=None,
                 drop_probability=0.0,
                 init='kaiming_normal'):
        """
        :param num_neurons: number of neurons for each layer
        :param out_activation: output layer's activation unit
        :param input_normalization: input normalization behavior flag
        0: Do not normalize, 1: Batch normalization, 2: Layer normalization
        :param hidden_activation: hidden layer activation units. supports 'relu','SELU','leaky_relu','sigmoid', 'tanh'
        :param init: hidden layer initialization. supports 'kaiming_normal'
        """

        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.out_activation = out_activation
        self.input_normalization = input_normalization
        self.hidden_activation = hidden_activation
        self.drop_probability = drop_probability
        self.init = init

        # infer normalization layers
        if self.input_normalization == 0:
            pass
        else:
            if self.input_normalization == 1:
                norm_layer = torch.nn.BatchNorm1d(self.input_dimension)
            elif self.input_normalization == 2:
                norm_layer = torch.nn.LayerNorm(self.input_dimension)
            self.layers.append(norm_layer)

        self.layers.append(torch.nn.Linear(self.input_dimension, num_neurons[0]))  # input -> hidden 1
        for i, num_neuron in enumerate(num_neurons[:-1]):
            hidden_layer = torch.nn.Linear(num_neuron, num_neurons[i + 1])
            self.apply_weight_init(hidden_layer, self.init)
            self.layers.append(hidden_layer)
        last_layer = torch.nn.Linear(num_neurons[-1], self.output_dimension)
        self.apply_weight_init(last_layer, self.init)
        self.layers.append(last_layer)  # hidden_n -> output

    def forward(self, x):
        if self.input_normalization != 0:  # The first layer is not normalization layer
            out = self.layers[0](x)
            for layer in self.layers[1:-1]:  # Linear layer starts from layers[1]
                out = layer(out)
                if self.drop_probability > 0.0:
                    out = self.infer_dropout(self.drop_probability)(out)  # Apply dropout
                out = self.infer_activation(self.hidden_activation)(out)

            out = self.layers[-1](out)  # The last linear layer
            if self.out_activation is None:
                pass
            else:
                out = self.infer_activation(self.out_activation)(out)
        else:
            out = x
            for layer in self.layers[:-1]:
                out = layer(out)
                if self.drop_probability > 0.0:
                    out = self.infer_dropout(self.drop_probability)(out)
                out = self.infer_activation(self.hidden_activation)(out)

            out = self.layers[-1](out)
            # infer output activation units
            if self.out_activation is None:
                pass
            else:
                out = self.infer_activation(self.out_activation)(out)
        return out

    def apply_weight_init(self, tensor, init_method=None):
        if init_method is None:
            pass  # do not apply weight init
        elif init_method == "normal":
            torch.nn.init.normal_(tensor.weight, std=0.3)
            torch.nn.init.constant_(tensor.bias, 0.0)
        elif init_method == "kaiming_normal":
            torch.nn.init.kaiming_normal_(tensor.weight, nonlinearity=self.hidden_activation)
            torch.nn.init.constant_(tensor.bias, 0.0)

    def infer_activation(self, activation):
        if activation == 'relu':
            ret = torch.nn.ReLU()
        elif activation == 'sigmoid':
            ret = torch.nn.Sigmoid()
        elif activation == 'SELU':
            ret = torch.nn.SELU()
        elif activation == 'leaky_relu':
            ret = torch.nn.LeakyReLU()
        elif activation == 'tanh':
            ret = torch.nn.Tanh()
        else:
            raise RuntimeError("Given {} activation is not supported".format(self.out_activation))
        return ret

    @staticmethod
    def infer_dropout(p):
        if p >= 0.0:
            ret = torch.nn.Dropout(p=p)
        return ret


class VAE(nn.Module):
    def __init__(self,
                 i=2,
                 lec=400,
                 lds=20,
                 ldec=400,
                 out=2,
                 ):
        '''
        :param in: input shape
        :param lec: latent vector size - encoding layer
        :param lds: latent distribution size (dimension of vectors mu, logvar)
        :param ldec: latent vector sizes - decoding layers
        0: Do not normalize, 1: Batch normalization, 2: Layer normalization
        :param out: output shape
        '''
        super(VAE, self).__init__()
        # input size
        self.i = i

        self.fc1 = nn.Linear(i, lec)
        self.fc21 = nn.Linear(lec, lds)
        self.fc22 = nn.Linear(lec, lds)

        self.fc3 = nn.Linear(lds, ldec)
        self.fc4 = nn.Linear(ldec, out)

    def encode(self, x):
        # encodes data x into mu, logvar
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.i))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar




