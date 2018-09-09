from script.model.sklearn_like_model.net_structure.ResNetStructure.BaseResNetStructure import BaseResNetStructure


class ResNet18Structure(BaseResNetStructure):
    def body(self, stacker):
        stacker.add_layer(self.residual_block_type2, self.n_channel)
        stacker.add_layer(self.residual_block_type2, self.n_channel)

        stacker.add_layer(self.residual_block_type2, self.n_channel * 2, down_sample=True)
        stacker.add_layer(self.residual_block_type2, self.n_channel * 2)

        stacker.add_layer(self.residual_block_type2, self.n_channel * 4, down_sample=True)
        stacker.add_layer(self.residual_block_type2, self.n_channel * 4)

        stacker.add_layer(self.residual_block_type2, self.n_channel * 8, down_sample=True)
        stacker.add_layer(self.residual_block_type2, self.n_channel * 8)

        return stacker
