from script.model.sklearn_like_model.NetModule.ResNetStructure.BaseResNetStructure import BaseResNetNetModule


class ResNet34NetModule(BaseResNetNetModule):
    def body(self, stacker):
        for i in range(3):
            stacker.add_layer(self.residual_block_type2, self.n_channel * 1)

        stacker.add_layer(self.residual_block_type2, self.n_channel * 2, down_sample=True)
        for i in range(4 - 1):
            stacker.add_layer(self.residual_block_type2, self.n_channel * 2)

        stacker.add_layer(self.residual_block_type2, self.n_channel * 4, down_sample=True)
        for i in range(6 - 1):
            stacker.add_layer(self.residual_block_type2, self.n_channel * 4)

        stacker.add_layer(self.residual_block_type2, self.n_channel * 8, down_sample=True)
        for i in range(3 - 1):
            stacker.add_layer(self.residual_block_type2, self.n_channel * 8)

        return stacker
