# graph-transformer-psiml
Transformer implemented with graph neural network attention layer from Pytorch Geometric. This was a project for [PSIML](https://psiml.petlja.org/), Practical Seminar for Machine Learning organized by PFE, Petlja, Everseen, and Microsoft in Belgrade 2022.

<div align="center">
  <img src="https://user-images.githubusercontent.com/34657562/184308361-554b6ce6-5cac-4f99-94c0-66bb48864d69.png" align="center" width="50%">
</div>

## Authors

- Marina Debogović (ETF)
- Marko Njegomir (FTN)

## Mentors
- Anđela Donević (Everseen)
- Nikola Popović (ETH Zurich)

<div align="center">
  <img src="https://user-images.githubusercontent.com/34657562/184306183-802cb780-29ce-4fed-95b6-82023b199354.png">
  <p align="center">Illustration 1 - Transformer with graph attention network (DALLE-2).</p>
</div>

# Architecture

- The attention layer in ViT Encoder is replaced with GATv2 (Graph Attention network).
  - Inputs for the GATv2 must be a single graph and an adjacency list.
      - To support batches, a disjoint union of graphs in the batch is created, so we get a single graph.
  - Output dim from the GATv2 is multiplied by the number of heads
      - A new layer is added that reduces the output dim to the input dimensions so the layers can be stacked.
- GATv2 layers can easily be replaced with any other GNN layer in Pytorch Geometric.
  - For some specific layers that take more than just vertices and edges some tweaks to the inputs and outputs might be necessary.

<div align="center">
  <img src="images/graph_transformer_encoder.png">
  <p align="center">Illustration 2 - Attention layer in Vision Transformer's Encoder is replaced with Graph Attention Network.</p>
</div>

# Results

- Trained and tested on VM with a single V100 GPU
- Due to time and hardware constraints, models were compared on MNIST and CIFAR10
- There were no pre-trained models on Imagenet with this architecture available, so no transfer learning was possible.
  - Training the model on Imagenet first and then finetuning to some other specific task might improve performance.

## MNIST

<div align="center">
  <img src="images/mnist-train-loss.png">
  <p align="center">Illustration 3 - MNIST train loss for Classic ViT and our Graph Transformer.</p>
</div>

<div align="center">
  <img src="images/mnist-train-acc.png">
  <p align="center">Illustration 4 - MNIST train accuracy for Classic ViT and our Graph Transformer.</p>
</div>

<div align="center">
  <img src="images/mnist-val-acc.png">
  <p align="center">Illustration 5 - MNIST validation accuracy for Classic ViT and our Graph Transformer.</p>
</div>

## CIFAR10

<div align="center">
  <img src="images/cifar10-train-loss.png">
  <p align="center">Illustration 6 - CIFAR10 train loss for Classic ViT and our Graph Transformer.</p>
</div>

<div align="center">
  <img src="images/cifar10-train-acc.png">
  <p align="center">Illustration 7 - CIFAR10 train accuracy for Classic ViT and our Graph Transformer.</p>
</div>

<div align="center">
  <img src="images/cifar10-val-acc.png">
  <p align="center">Illustration 8 - CIFAR10 validation accuracy for Classic ViT and our Graph Transformer.</p>
</div>
