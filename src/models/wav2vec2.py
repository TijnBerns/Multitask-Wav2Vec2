from transformers import Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
from config import Config
import torch


def load_processor() -> Wav2Vec2Processor:
    # Initialize processor
    tokenizer = Wav2Vec2CTCTokenizer(
        "src/models/vocab.json", unk_token="<unk>", pad_token="<pad>", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=Config.sample_rate, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer)
    return processor


def load_model() -> Wav2Vec2ForCTC:
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-base-960h", ctc_loss_reduction="mean")
    model.config.vocab_size = 33
    model.lm_head = torch.nn.Linear(in_features=768, out_features=33)
    model.freeze_feature_encoder()
    return model
