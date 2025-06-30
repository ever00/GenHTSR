# GenHTSR  
Generative Handwritten Text Separation and Recognition using a Diffusion Model, Pix2Pix, and Attention-based HTR.

### Synthetic Palimpsest Samples

Three **synthetic palimpsests** are generated in this project, showcasing different levels of complexity:

- `generate_mnist_palimpsests.ipynb` — a straightforward dataset with overlapping elements utilizing MNIST [^1] as under-text.  
- `generate_saintgall_palimpsests.ipynb` — a more challenging version with historical handwritten samples from the Saint Gall dataset [^2] as under-text.  
- `generate_georgian_patches.py` — a processed version of the public synthetic Georgian dataset [^3] simulating the real Georgian palimpsest [^4].

These samples are used to evaluate the separation and recognition performance of the proposed method.

### Generative Text Separation Networks

- **pix2pix** — a conditional generative adversarial network.  
- **cDDM** — a custom denoising diffusion model, inspired by DDPM but with a modified noise mechanism targeting structured noise.

### Recognition Network

The recognition network used in this project is the pre-trained **AttentionHTR** by Kass and Vats [^5].

- **AttentionHTR GitHub**: [https://github.com/dmitrijsk/AttentionHTR.git](https://github.com/dmitrijsk/AttentionHTR.git)

### References

[^1]: Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner.  
Gradient-based learning applied to document recognition.  
*Proceedings of the IEEE*, 86(11):2278–2324, 1998.  
[Link to paper](https://ieeexplore.ieee.org/document/726791)

[^2]: Andreas Fischer, Volker Frinken, Christopher Kwigl, and Horst Bunke.  
Transcription alignment of Latin manuscripts using hidden Markov models.  
In *Proceedings of the 15th International Conference on Document Analysis and Recognition (ICDAR)*, pages 754–758, 2011.

[^3]: Mahdi Jampour, Hussein Mohammed, and Jost Gippert.  
Enhancing the readability of palimpsests using generative image inpainting.  
In *International Conference on Pattern Recognition Applications and Methods*, 2024.

[^4]: UCLA Sinai Library Project.  
About the project, 2025.  
[http://sinaipalimpsests.org/](http://sinaipalimpsests.org/)

[^5]: Dmitrijs Kass and Ekta Vats.  
*AttentionHTR: Handwritten Text Recognition Based on Attention Encoder-Decoder Networks.*  
In *Proceedings of the International Workshop on Document Analysis Systems*, Springer International Publishing, 2022.  
[arXiv:2201.09390](https://arxiv.org/abs/2201.09390)
