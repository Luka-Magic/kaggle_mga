
import torch.nn as nn

from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention

class CRNN(nn.Module):
    def __init__(self, cfg, n_chars):
        super().__init__()
        self.cfg = cfg
        self.stages = {'Trans': cfg.Transformation, 'Feat': cfg.FeatureExtraction,
                       'Seq': cfg.SequenceModeling, 'Pred': cfg.Prediction}

        """ Transformation """
        if cfg.Transformation == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=cfg.num_fiducial, I_size=(cfg.img_h, cfg.img_w), I_r_size=(cfg.img_h, cfg.img_w), I_channel_num=cfg.input_channel)
        else:
            print('No Transformation module specified')

        """ FeatureExtraction """
        if cfg.FeatureExtraction == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(cfg.input_channel, cfg.output_channel)
        elif cfg.FeatureExtraction == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(cfg.input_channel, cfg.output_channel)
        elif cfg.FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(cfg.input_channel, cfg.output_channel)
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = cfg.output_channel  # int(imgH/16-1) * 512
        if cfg.FeatureExtraction == 'EfficientNet':
            self.FeatureExtraction_output = 1280  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        if cfg.SequenceModeling == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, cfg.hidden_channel, cfg.hidden_channel),
                BidirectionalLSTM(cfg.hidden_channel, cfg.hidden_channel, cfg.hidden_channel))
            self.SequenceModeling_output = cfg.hidden_channel
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if cfg.Prediction == 'CTC':
            self.Prediction = nn.Linear(self.SequenceModeling_output, n_chars)
        elif cfg.Prediction == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, cfg.hidden_channel, n_chars)
        else:
            raise Exception('Prediction is neither CTC or Attn')

    def forward(self, input, text, is_train=True):
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        """ Prediction stage """
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.cfg.batch_max_length)

        return prediction