from typing import Tuple

from torch import nn
import torch.nn.functional as F

class ControlNetConditioningEmbedding(nn.Module):


    def __init__(
        self,
        cond_dim,
        embed_dim
    ):
        super().__init__()

        # self.conv_in = nn.Conv1d(conditioning_channels, block_out_channels[0],
        #                          kernel_size=block_strides[0] + 1, stride=block_strides[0])

        self.blocks = nn.ModuleList([])

        # for i in range(len(block_out_channels) - 1):
        #    ...
        #    channel_in = block_out_channels[i]
        #    channel_out = block_out_channels[i + 1]
        #    self.blocks.append(nn.Conv1d(channel_in, channel_in, kernel_size=3, padding=1))
        #    self.blocks.append(nn.Conv1d(channel_in, channel_out, kernel_size=block_strides[i + 1]+1,
        #                                 padding=1, stride=block_strides[i + 1]))
        #self.conditioning_embedding_channels = conditioning_embedding_channels

    def forward(self, cond):
        return cond
        # embedding = self.conv_in(cond)
        # embedding = F.silu(embedding)

        # for block in self.blocks:
        #     embedding = block(embedding)
        #     embedding = F.silu(embedding)
        # embedding = embedding.reshape(embedding.shape[0],
        #                               self.conditioning_embedding_channels,
        #                               embedding.shape[1] // self.conditioning_embedding_channels,
        #                               embedding.shape[2])
        # embedding = embedding.permute(0, 1, 3, 2)
        # return embedding