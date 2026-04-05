import torch.nn as nn
import fvdb
import fvdb.nn as fvnn


class Encoder(nn.Module):
    def __init__(self, features=256, n_layers=10):
        super(Encoder, self).__init__()
        self.activation = fvnn.SiLU(inplace=True)
        net = [
            fvnn.SparseConv3d(1, features, kernel_size=3, stride=1),
            fvnn.BatchNorm(features),
            self.activation]
        for i in range(n_layers):
            net += [
                fvnn.SparseConv3d(features, features, kernel_size=3, stride=1),
                fvnn.BatchNorm(features),
                self.activation]
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, features=256, n_layers=10):
        super(Decoder, self).__init__()
        self.activation = fvnn.SiLU(inplace=True)
        net = [
            fvnn.SparseConv3d(2*features, features, kernel_size=1, stride=1),
            fvnn.BatchNorm(features),
            self.activation]
        for i in range(n_layers):
            net += [
                fvnn.SparseConv3d(features, features, kernel_size=1, stride=1),
                fvnn.BatchNorm(features),
                self.activation]
        net += [fvnn.SparseConv3d(features, 1, kernel_size=1, stride=1)]
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)

# 128,5
# 256,5

class EncoderDecoder(nn.Module):
    def __init__(self, features=256, n_layers=10):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(features, n_layers)
        self.decoder = Decoder(features, n_layers)
        self.pos_encoding = fvnn.SparseConv3d(
            3, features, kernel_size=1, stride=1)

    def forward(self, x):
        sdf_input = fvnn.VDBTensor(
            x.grid, x.grid.jagged_like(x.jdata[:, :1].contiguous()))
        pos_input = fvnn.VDBTensor(
            x.grid, x.grid.jagged_like(x.jdata[:, 1:].contiguous()))
        pos_input = self.pos_encoding(pos_input)
        enc_out = self.encoder(sdf_input)
        dec_out = self.decoder(fvdb.jcat([enc_out, pos_input], dim=1))
        return dec_out
