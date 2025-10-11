import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class CNN_LSTM_Model(nn.Module):
    """
    A hybrid CNN-LSTM model for spatio-temporal feature extraction.
    The model first uses 3D convolutional blocks to extract spatial features
    from the input, then flattens the output and passes it to an LSTM to
    model temporal dependencies.
    """
    def __init__(self, X_train_shape):
        super(CNN_LSTM_Model, self).__init__()

        # CNN Feature Extractor
        self.conv_block1 = nn.Sequential(OrderedDict([
            ('bn1', nn.BatchNorm3d(num_features=X_train_shape[1])),
            ('conv1', nn.Conv3d(
                in_channels=1,
                out_channels=16,
                kernel_size=(int(X_train_shape[2] / 2), 5, 5),
                stride=(1, 2, 2)
            )),
            ('relu1', nn.ReLU()),
            ('pool1', nn.MaxPool3d(kernel_size=2, stride=2))
        ]))

        self.conv_block2 = nn.Sequential(OrderedDict([
            ('bn2', nn.BatchNorm3d(16)),
            ('conv2', nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1)),
            ('relu2', nn.ReLU()),
            ('pool2', nn.AvgPool3d(kernel_size=3, stride=2))
        ]))

        # Dynamically determine the input size for the LSTM
        lstm_input_size = self._get_lstm_input_size(X_train_shape)

        # LSTM and Classifier
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=512, num_layers=3, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def _get_lstm_input_size(self, X_train_shape):
        """Calculates the input feature dimension for the LSTM layer."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *X_train_shape[1:])
            x = self.conv_block1(dummy_input)
            x = self.conv_block2(x)
            x = x.squeeze(2)
            x = torch.flatten(x, 1)
            return x.shape[1]

    def _log_shape(self, tensor, name, verbose):
        """Helper function to print tensor shapes if verbose is True."""
        if verbose:
            print(f'Shape after {name}: {tensor.shape}')

    def forward(self, x, verbose=False):
        self._log_shape(x, 'input', verbose)
        
        # Apply convolutional blocks
        x = self.conv_block1(x)
        self._log_shape(x, 'conv_block1', verbose)
        
        x = self.conv_block2(x)
        self._log_shape(x, 'conv_block2', verbose)

        # Prepare for LSTM
        x = x.squeeze(2)
        x = torch.flatten(x, 1)
        self._log_shape(x, 'flatten', verbose)
        
        # LSTM layer
        x, _ = self.lstm(x)
        self._log_shape(x, 'LSTM', verbose)
        
        # Classifier
        x = self.classifier(x)
        self._log_shape(x, 'classifier', verbose)
        
        return x


class PatchEmbedding(nn.Module):
    """
    Converts a 2D image into a sequence of flattened patch embeddings.
    """
    def __init__(self, patch_size, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # Project image into patches and flatten the spatial dimensions
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class TransformerEncoder(nn.Module):
    """
    A standard Transformer encoder block.
    """
    def __init__(self, embed_dim, num_heads, hidden_dim, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, hidden_dim, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x):
        return self.transformer_encoder(x)


class MViT(nn.Module):
    """
    Multi-channel Vision Transformer (MViT).
    This model processes each channel of a multi-channel input independently
    through its own Transformer encoder, then concatenates the results for
    final classification.
    """
    def __init__(self, X_shape, in_channels, num_classes, patch_size,
                 embed_dim, num_heads, hidden_dim, num_layers, dropout):
        super(MViT, self).__init__()

        # Create a set of modules for each input channel
        self.channel_processors = nn.ModuleList([
            self._create_channel_processor(X_shape, patch_size, embed_dim, num_heads, hidden_dim, num_layers, dropout)
            for _ in range(in_channels)
        ])
        
        self.head = nn.Linear(embed_dim * in_channels, num_classes)

    def _create_channel_processor(self, X_shape, patch_size, embed_dim, num_heads, hidden_dim, num_layers, dropout):
        """Helper to create the patch embedding, pos embedding, CLS token, and encoder for a single channel."""
        num_patches = (X_shape[3] // patch_size[0]) * (X_shape[4] // patch_size[1])
        return nn.ModuleDict({
            'patch_embed': PatchEmbedding(patch_size, embed_dim),
            'pos_embed': nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim)),
            'cls_token': nn.Parameter(torch.zeros(1, 1, embed_dim)),
            'encoder': TransformerEncoder(embed_dim, num_heads, hidden_dim, num_layers, dropout)
        })

    def _process_channel(self, channel_input, processor):
        """Forward pass for a single channel."""
        embedded_patches = processor['patch_embed'](channel_input)
        batch_size = embedded_patches.size(0)
        
        # Prepend CLS token
        cls_tokens = processor['cls_token'].expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, embedded_patches), dim=1)
        
        # Add positional embedding
        x += processor['pos_embed']
        
        # Pass through Transformer encoder
        encoded_output = processor['encoder'](x)
        
        # Return the CLS token's output
        return encoded_output[:, 0]

    def forward(self, x):
        # Assuming input shape (Batch, Channels, Time=1, Height, Width)
        # Squeeze the time dimension
        x = x.squeeze(2)
        
        # Process each channel independently
        channel_outputs = [
            self._process_channel(x[:, i:i+1, :, :], self.channel_processors[i])
            for i in range(x.size(1))
        ]
        
        # Concatenate the CLS token outputs from all channels
        concatenated_output = torch.cat(channel_outputs, dim=1)
        
        # Final classification head
        return self.head(concatenated_output)