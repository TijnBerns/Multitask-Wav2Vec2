
from numpy import NaN
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torchaudio
import torch
from config import Config
import data
from tqdm import tqdm


def train(dataloader, model: Wav2Vec2ForCTC, processor, num_epochs, device):
    optim = torch.optim.Adam(model.parameters(), lr=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, num_epochs)
    
    for _ in range(num_epochs):
        epoch(dataloader, model, processor, optim, device)
        scheduler.step()


def epoch(dataloader, model, processor, optim, device):
    total_loss = 0
    
    for sample in tqdm(dataloader):
        # Skip the sample if the transcription is longer than some threshold
        if any(len(x) > 180  for x in sample['transcription']):
            continue
        
        input_values = processor(
            sample['waveform'], sampling_rate=16_000, return_tensors="pt", padding="longest").input_values
        target_transcription = sample['transcription']

        # Retrieve input values and target labels
        input_values = torch.reshape(input_values, (1, -1))
        with processor.as_target_processor():
            labels = processor(target_transcription,
                               return_tensors='pt').input_ids


        # Compute loss by passing labels
        output = model(input_values, labels=labels)
        loss = output.loss
        optim.zero_grad()
        loss.backward()
        optim.step()

        # Print prediction vs target
        logits = output.logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        print(f"Target transcription:   {target_transcription}\n" +
              f"Prediction:             {transcription}\n" + 
              f"Loss:                   {loss}\n" + 
              f"predicted_ids:          {predicted_ids}")
        
    return total_loss / len(dataloader.dataset)


if __name__ == "__main__":
    # Load model and processor
    processor = Wav2Vec2Processor.from_pretrained(
        pretrained_model_name_or_path="facebook/wav2vec2-base-960h", 
        vocab_file='/home/tijn/CS/Master/Speaker_Change_Recognition/models/vocab.json')
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h", vocab_size=32)
    

    # load dummy dataset and read soundfiles
    # dev_clean_loader = data.get_loaders(Config)
    dataset = data.CustomLibriSpeechDataset(Config.datapath + '/test')
    # dev_ds = torchaudio.datasets.LIBRISPEECH(
    #     Config.datapath, url="dev-clean", download=True)

    loader = data.initialize_loader(Config, dataset)

    # perform training
    train(loader, model, processor, 10, 'cpu')


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
