













Train a ResNet-18 model with knowledge distilled from a pre-trained ResNext-29 teacher

```
python3 train.py --model_dir experiments/resnet18_distill/resnext_teacher
```

+ Test accuracy: **94.788%**



## References

Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. "Distilling the knowledge in a neural network." arXiv preprint arXiv:1503.02531 (2015).

Romero, A., Ballas, N., Kahou, S. E., Chassang, A., Gatta, C., & Bengio, Y. (2014). Fitnets: Hints for thin deep nets. arXiv preprint arXiv:1412.6550.
