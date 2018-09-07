from script.model.sklearn_like_model.net_structure.ResNetStructure.BaseResNetStructure import BaseResNetStructure


class ResNet152Structure(BaseResNetStructure):
    def body(self, stacker):
        for i in range(3):
            stacker.add_layer(self.residual_block_type3, 64)

        stacker.add_layer(self.residual_block_type3, 128, down_sample=True)
        for i in range(8 - 1):
            stacker.add_layer(self.residual_block_type3, 128)

        stacker.add_layer(self.residual_block_type3, 256, down_sample=True)
        for i in range(36 - 1):
            stacker.add_layer(self.residual_block_type3, 256)

        stacker.add_layer(self.residual_block_type3, 512, down_sample=True)
        for i in range(3):
            stacker.add_layer(self.residual_block_type3, 512)

        return stacker
