# kaggle_mga

## 20230326 時点での全体フロー

0. [create dataset](./0_create_dataset/)

    - key point detection用のkey pointを生成

    - imageとjsonをlmdb化して取り出しやすくした。

1. [chart classification](./1_chart_classification/)

    - チャートを5種類に分類

2. [key point detection](./2_key_point_detection/)

    - key point detection

3. [text detection](./3_text_detection/)

    - CRAFTを使ってdetection

    - 実データでfine tuning

4. [text recognition](./4_text_recognition/)

    - CRNNで文字認識モデルを学習

    - fine tuning(元あったっけ)

5. [text role classification](./5_text_role_classification/)

    - 3, 4の検出値からx, y-ticksに分類されるものを見つける