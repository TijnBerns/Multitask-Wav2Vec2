
from numpy import NaN
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor
import torch
from config import Config
import data
from tqdm import tqdm
import math


def train(dataloader, model: Wav2Vec2ForCTC, processor, num_epochs, device):
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, num_epochs)
    print("epoch\tloss")
    for i in range(num_epochs):
        loss = epoch(dataloader, model, processor, optim, device)
        scheduler.step()
        print(f"{i}\t{loss}")


def epoch(dataloader, model, processor, optim, device):
    total_loss = 0

    for i, sample in tqdm(enumerate(dataloader)):
        # Skip the sample if the transcription is longer than some threshold
        # if any(len(x) > 300 for x in sample['transcription']):
        #     continue
        waveform = sample['waveform'].to(device)

        input_values = processor(
            waveform, sampling_rate=16_000, return_tensors="pt", padding="longest").input_values
        target_transcription = sample['transcription']

        # Retrieve input values and target labels
        input_values = torch.reshape(
            input_values, (Config.batch_size, -1)).to(device)
        with processor.as_target_processor():
            labels = processor(target_transcription, padding=True,
                               return_tensors='pt').input_ids
            labels = labels.to(device)

        # Compute loss by passing labels
        output = model(input_values, labels=labels)
        loss = output.loss

        if math.isnan(loss):
            exit("LOSS REACHED NAN VALUE")

        optim.zero_grad()
        loss.backward()
        optim.step()

        # Print prediction vs target
        if i % 100 == 0:
            logits = output.logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)
            print(f"Target transcription:   {target_transcription}\n" +
                  f"Prediction:             {transcription}\n" +
                  f"Loss:                   {loss}\n" +
                  f"Labels:                 {labels}\n")

    return total_loss / len(dataloader.dataset)


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available else 'cpu'

    # Load model and processor
    tokenizer = Wav2Vec2CTCTokenizer(
        "models/vocab.json", unk_token="<unk>", pad_token="<pad>", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=Config.sample_rate, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer)

    # processor = Wav2Vec2Processor.from_pretrained(
    #     pretrained_model_name_or_path="facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-base-960h", ctc_loss_reduction="mean", pad_token_id=processor.tokenizer.pad_token_id).to(device)
    model.freeze_feature_encoder()

    # load dummy dataset and read soundfiles
    # dev_clean_loader = data.get_loaders(Config)
    dataset = data.CustomLibriSpeechDataset(Config.datapath + '/test')
    # dataset = torchaudio.datasets.LIBRISPEECH(
    #     Config.datapath, url="dev-clean", download=True)

    loader = data.initialize_loader(Config, dataset)

    # perform training
    train(loader, model, processor, 10, device)


"""
for sample in dev_clean_loader:

    input_values = processor(sample['waveform'], sampling_rate=16_000, return_tensors="pt", padding="longest").input_values  
    
    # retrieve logits
    input_values = torch.reshape(input_values, (1,-1))
    logits = model(input_values).logits
    

    # take argmax and decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    
    breakpoint()
        
"""
