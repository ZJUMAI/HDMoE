# HDMoE
- [x] 2026-05-17 ⭐: Congratulations! 🎉🎊🎉 HDMoE has been accepted by KDD 2026 🎯
<summary>
  <b>HDMoE: A Hierarchical Decoupling-Fusion Mixture-of-Experts Framework for Multimodal Cancer Survival Prediction</b>.
  
  <br><em>Huayi Wang, Haochao Ying, Yuyang Xu, Qiyao Zheng, jun wang, Cheng Zhang, Ying Sun, Jian Wu </em></br>
</summary>

<img src="imgs/framework.png" width="1500px" align="center" />

</details>
**Summary:** Here is the official implementation of the paper "HDMoE: A Hierarchical Decoupling-Fusion Mixture-of-Experts Framework for Multimodal Cancer Survival Prediction".




### Pre-requisites:
```bash
torch 2.3.1+cu121
scikit-survival 0.23.0
```

### Prepare your data
#### WSIs
1. Download diagnostic WSIs from [TCGA](https://portal.gdc.cancer.gov/)
2. Use the WSI processing tool provided by [CLAM](https://github.com/mahmoodlab/CLAM) to extract resnet-50 pretrained 1024-dim feature for each 256 $\times$ 256 patch (20x), which we then save as `.pt` files for each WSI. So, we get one `pt_files` folder storing `.pt` files for all WSIs of one study.

The final structure of datasets should be as following:
```bash
DATA_ROOT_DIR/
    └──pt_files/
        ├── slide_1.pt
        ├── slide_2.pt
        └── ...
```

DATA_ROOT_DIR is the base directory of cancer type (e.g. the directory to TCGA_BLCA), which should be passed to the model with the argument `--data_root_dir` as shown in [run.sh](run.sh).

#### Genomics
In this work, we directly use the preprocessed genomic data、signatures.csv and sample lists (Training-Validation Splits) provided by [PORPOISE](https://github.com/mahmoodlab/PORPOISE), stored in folder [csv](./csv) and [splits](./splits).

## Training-Validation Splits
Splits for each cancer type are found in the `splits/5foldcv ` folder, which are randomly partitioned each dataset using 5-fold cross-validation. Each one contains splits_{k}.csv for k = 1 to 5. 

## Running Experiments
To train HDMoE, you can specify the argument in the bash `run.sh` and run the command:
```bash
bash run.sh
```
or use the following generic command-line and specify the arguments:
```bash
CUDA_VISIBLE_DEVICES=<DEVICE_ID> python main.py \
                                      --which_splits 5foldcv \
                                      --dataset <CANCER_TYPE> \
                                      --data_root_dir <DATA_ROOT_DIR>\
                                      --modal coattn \
                                      --model hdmoe \
                                      --num_epoch 30 \
                                      --batch_size 1 \
                                      --loss nll_surv_cos \
                                      --lr 0.0005 \
                                      --optimizer Adam \
                                      --scheduler None \
                                      --alpha 1.0
```


## Acknowledgements
Huge thanks to the authors of following open-source projects:
- [CLAM](https://github.com/mahmoodlab/CLAM)

## License & Citation 
If you find our work useful in your research, please consider citing our paper at:
```bash
@ARTICLE{HDMoE,
  author={Huayi Wang and Haochao Ying and Yuyang Xu and Qiyao Zheng and jun wang and Cheng Zhang and Ying Sun and Jian W},
  title={HDMoE: A Hierarchical Decoupling-Fusion Mixture-of-Experts Framework for Multimodal Cancer Survival Prediction}, 
  year={2026},
  journal={arXiv preprint arXiv:2605.20891},
  keywords={Multimodal Learning; Mixture-of-Experts; Survival Prediction},
  }

