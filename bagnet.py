from bagnets.pytorchnet import BagNet, Bottleneck
from torch import Tensor


class ClippedBagNet(BagNet):
    def __init__(
        self,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        strides=[2, 2, 2, 1],
        kernel3=[1, 1, 1, 1],
        num_classes=1000,
        clip_range=None,
        aggregation='cbn',
        avg_pool=False,
        alpha=0.05,
        beta=-1,
    ):
        super(ClippedBagNet, self).__init__(
            block,
            layers,
            strides=strides,
            kernel3=kernel3,
            num_classes=num_classes,
            avg_pool=avg_pool,
        )
        self.clip_range = clip_range
        self.aggregation = aggregation
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes

    def forward(self, x, y=None):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.permute(0, 2, 3, 1)
        x = self.fc(x)

        if self.clip_range is not None:
            x = Tensor.clamp(x, self.clip_range[0], self.clip_range[1])

        if self.aggregation == 'mean':
            x = Tensor.mean(x, dim=(1, 2))

        elif self.aggregation == 'median':
            x = x.view([x.size()[0], -1, 10])
            x = Tensor.median(x, dim=1)
            return x.values

        elif self.aggregation == 'cbn':  # clipped BagNet
            x = Tensor.tanh(x * self.alpha + self.beta)
            x = Tensor.mean(x, dim=(1, 2))

        elif self.aggregation == 'none':
            pass

        return x
