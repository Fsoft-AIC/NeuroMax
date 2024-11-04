# Code for NeuroMax: Enhancing Neural Topic Modeling via Maximizing Mutual Information and Group Topic Regularization (EMNLP 2024 Findings)

[Paper link](https://arxiv.org/abs/2409.19749)

## Preparing libraries
1. Install the following libraries
    ```
    numpy 1.26.4
    torch_kmeans 0.2.0
    pytorch 2.2.0
    sentence_transformers 2.2.2
    scipy 1.10
    bertopic 0.16.0
    gensim 4.2.0
    ```
2. Install java
3. Download [this java jar](https://hobbitdata.informatik.uni-leipzig.de/homes/mroeder/palmetto/palmetto-0.1.0-jar-with-dependencies.jar) to ./evaluations/pametto.jar for evaluating
4. Download and extract [this processed Wikipedia corpus](https://hobbitdata.informatik.uni-leipzig.de/homes/mroeder/palmetto/Wikipedia_bd.zip) to ./datasets/wikipedia/ as an external reference corpus.

## Usage
To run and evaluate our model for YahooAnswers dataset, run this example:

> python main.py --use_pretrainWE

## Acknowledgement
Some part of this implementation is based on [TopMost](https://github.com/BobXWu/TopMost). We also utilizes [Palmetto](https://github.com/dice-group/Palmetto) for the evaluation of topic coherence.

## Citation

If you want to reuse our code, please cite us as:

```
@misc{pham2024neuromax,
      title={NeuroMax: Enhancing Neural Topic Modeling via Maximizing Mutual Information and Group Topic Regularization}, 
      author={Duy-Tung Pham and Thien Trang Nguyen Vu and Tung Nguyen and Linh Ngo Van and Duc Anh Nguyen and Thien Huu Nguyen},
      year={2024},
      eprint={2409.19749},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.19749}, 
}
```