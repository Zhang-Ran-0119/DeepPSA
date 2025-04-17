# DeepPSA: A Geometric Deep Learning Model for PROTAC Synthetic Accessibility Prediction
PROTACs have garnered significant attention in drug design due to their ability to induce the degradation of the target proteins via ubiquitin-proteasome system. However, the synthesis of PROTACs remains a challenging process, requiring the consideration of factors such as chemical complexity and accessibility. With the rise of generative artificial intelligence, several PROTAC generation models have been introduced, but tools to evaluate the synthetic accessibility of these molecules remain underdeveloped. 

To address this gap, we propose a deep learning-based computational model named DeepPSA (PROTAC Synthetic Accessibility), designed to predict the synthetic accessibility of PROTACs. You can use DeepSA on a webserver at [https://bailab.siais.shanghaitech.edu.cn/deepsa](https://bailab.siais.shanghaitech.edu.cn/services/psa)

## Prepare the environment
Here we export our anaconda environment as the file "environment.yml". You can use the command:
```
 conda env create -f environment.yml
 conda activate PSA
```
## Data
The original data and processed data can be obtained from the following links:
https://drive.google.com/drive/folders/1JaaxdXz65unBQWOosVU-q0DxPHNLQTnG

## Usage For Researchers

If you want to train your own model, you can run it from the command line,
running:
```
python PSA_finetuning.py <model_name> <train.csv> <valid.csv> <test.csv>
```
If you want to use the model we proposed,
running:
```
python PSA.py <data.csv>
```

## Online Server
We deployed a pre-trained model on a dedicated server, which is publicly available at [https://bailab.siais.shanghaitech.edu.cn/services/psa], to make it easy for biomedical researcher users to utilize DeepSA in their research activity.

Users can upload their SMILES or csv files to the server, and then they can quickly obtain the predicted results.

## Contact
If you have any questions, please feel free to contact Ran Zhang (Email: zhangran2023@shanghaitech.edu.cn).

Pull requests are highly welcomed!

## Acknowledgements
We are grateful for the support from HPC Platform of ShanghaiTech University.
Thank you all for your attention to this work.
