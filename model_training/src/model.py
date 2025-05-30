# src/model.py
# Definiert die Architektur des Custom Convolutional Neural Network (CNN).
# Das Modell ist flexibel gestaltet, um verschiedene Konfigurationen (Anzahl Blöcke, Filter, Dropout)
# basierend auf Hyperparametern zu ermöglichen, die z.B. von Optuna vorgeschlagen werden.

import torch
import torch.nn as nn


class CustomCNN(nn.Module):
    """
    Eine benutzerdefinierte CNN-Architektur für die binäre Bildklassifikation.
    Die Architektur besteht aus mehreren Convolutional Blocks, gefolgt von
    Global Average Pooling und einer Fully Connected Schicht für die Ausgabe.
    """

    def __init__(self, num_classes: int = 1, dropout_rate: float = 0.5,
                 num_conv_blocks: int = 4, first_layer_filters: int = 64,
                 filter_increase_factor: float = 2.0):
        """
        Initialisiert das CustomCNN Modell.

        Args:
            num_classes (int): Anzahl der Ausgabeklassen (1 für binäre Klassifikation mit BCEWithLogitsLoss).
            dropout_rate (float): Dropout-Wahrscheinlichkeit für den Fully Connected Layer.
            num_conv_blocks (int): Anzahl der zu stapelnden Convolutional Blocks.
            first_layer_filters (int): Anzahl der Filter (Ausgabekanäle) im ersten Convolutional Block.
            filter_increase_factor (float): Faktor, um den die Anzahl der Filter in nachfolgenden
                                            Convolutional Blocks erhöht wird.
        """
        super(CustomCNN, self).__init__()

        self.conv_blocks = nn.ModuleList()  # Eine Liste, um die Convolutional Blocks zu speichern
        current_channels = 3  # Eingabekanäle (RGB-Bilder)
        next_channels = int(first_layer_filters)  # Sicherstellen, dass es ein Integer ist

        # Erzeuge die Convolutional Blocks dynamisch
        for i in range(int(num_conv_blocks)):  # Sicherstellen, dass es ein Integer ist
            block = nn.Sequential(
                nn.Conv2d(current_channels, next_channels,
                          kernel_size=3, stride=1, padding='same', bias=False),  # 3x3 Kernel, 'same' Padding
                nn.BatchNorm2d(next_channels),  # Batch Normalization zur Stabilisierung
                nn.ReLU(inplace=True),  # ReLU Aktivierungsfunktion
                nn.MaxPool2d(kernel_size=2, stride=2)  # Max Pooling zur Dimensionsreduktion
            )
            self.conv_blocks.append(block)
            current_channels = next_channels  # Ausgabekanäle dieses Blocks sind Eingabekanäle des nächsten

            # Erhöhe die Anzahl der Kanäle für den nächsten Block,
            # außer es ist der letzte Convolutional Block.
            if i < int(num_conv_blocks) - 1:
                next_channels = int(current_channels * filter_increase_factor)
                next_channels = max(next_channels, 1)  # Mindestens 1 Ausgabekanal sicherstellen

        # Klassifikationskopf
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling reduziert jede Feature Map auf 1x1
        self.flatten = nn.Flatten()  # Flacht den Output für den FC-Layer ab
        self.dropout = nn.Dropout(p=float(dropout_rate))  # Sicherstellen, dass es ein Float ist

        # Der Fully Connected Layer.
        # Die Anzahl der Eingabefeatures ist die Anzahl der Kanäle des letzten Conv-Blocks,
        # da Global Average Pooling die räumliche Dimension auf 1x1 reduziert.
        self.fc = nn.Linear(current_channels, num_classes)

        self._initialize_weights()  # Initialisiere die Gewichte des Modells

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Definiert den Forward Pass des Modells.

        Args:
            x (torch.Tensor): Der Eingabe-Tensor (Batch von Bildern).

        Returns:
            torch.Tensor: Der Ausgabe-Tensor des Modells (Logits für binäre Klassifikation).
        """
        for block in self.conv_blocks:
            x = block(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def _initialize_weights(self) -> None:
        """
        Initialisiert die Gewichte der Convolutional und Linear Layer.
        Verwendet Kaiming Normal für Conv2d und eine kleine Normalverteilung für Linear.
        BatchNorm Gewichte werden auf 1 gesetzt, Bias auf 0.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # bias=False im Conv2d, daher keine Bias-Initialisierung hier nötig
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)  # Kleine Initialisierung für FC-Gewichte
                if m.bias is not None:  # Überprüfe, ob der Layer einen Bias hat
                    nn.init.constant_(m.bias, 0)