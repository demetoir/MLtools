from script.model.sklearn_like_model.net_structure.ResNetStructure.BaseResNetStructure import BaseResNetStructure


class ResNet18Structure(BaseResNetStructure):
    def body(self, stacker):
        stacker.add_layer(self.residual_block_type2, 64)
        stacker.add_layer(self.residual_block_type2, 64)

        stacker.add_layer(self.residual_block_type2, 128, down_sample=True)
        stacker.add_layer(self.residual_block_type2, 128)

        stacker.add_layer(self.residual_block_type2, 256, down_sample=True)
        stacker.add_layer(self.residual_block_type2, 256)

        stacker.add_layer(self.residual_block_type2, 512, down_sample=True)
        stacker.add_layer(self.residual_block_type2, 512)

        return stacker
