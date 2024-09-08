import torch
import numpy as np
import pandas as pd
import os
from torch import nn
from torch import optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

# Load the dataframe containing file names and labels
df = pd.read_csv('../new_codes_balanced_datasets/csv_file_generation_code_csv/balanced_text_label_file_iemocap.csv')

def plot_tsne(embeddings, labels, epoch, filename):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(16, 10))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, ticks=range(4), label='Emotion Labels')
    plt.title(f't-SNE of Embeddings at Epoch {epoch}')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.savefig(filename)
    plt.close()

# Function to load embeddings from files
def load_embeddings(file_names, folder_path, is_audio=False):
    embeddings = []
    for file_name in file_names:
        file_path = f"{folder_path}/{file_name}.npy"
        embedding = np.load(file_path)
        if is_audio:
            embedding = embedding.T  # Transpose the audio embedding to match the text embedding shape
            embedding = embedding[np.newaxis, :, :]  # Add batch dimension to become (1, time step, 20)
            embedding = embedding.squeeze(0)
        embeddings.append(torch.tensor(embedding, dtype=torch.float32))
    return embeddings if is_audio else torch.stack(embeddings)

# Function to extract gender from file names
def extract_gender(file_names):
    genders = []
    for name in file_names:
        # Split the filename and extract the last segment
        gender_code = name.split('_')[-1]
        # Check the first character of the last segment
        if gender_code[0] == 'F':
            genders.append(0)  # Female
        elif gender_code[0] == 'M':
            genders.append(1)  # Male
        else:
            print(f"Unexpected gender code in filename: {name}")
            genders.append(-1)  # For unexpected cases
    return torch.tensor(genders, dtype=torch.long)

# Enhanced Contrastive Loss for Batch Operations
class BatchContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(BatchContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        # Calculate pairwise distances
        distance_matrix = torch.cdist(embeddings, embeddings, p=2)
        # print(f"distance matrix:")
        # print(distance_matrix)

        # Create label matrix
        labels = labels.unsqueeze(1)
        label_matrix = (labels == labels.T).float()

        # Calculate losses
        positive_loss = label_matrix * distance_matrix**2
        negative_loss = (1 - label_matrix) * torch.clamp(self.margin - distance_matrix, min=0.0)**2

        # Average the losses
        loss = positive_loss + negative_loss
        loss = torch.sum(loss) / (len(labels) * len(labels) - len(labels))  # Avoid diagonal elements
        # print("*****************")
        # print(f"Contrastive loss: {loss}")
        return loss

# Split the data into train, validation, and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42, shuffle=True)

# Load text and audio embeddings
text_embeddings_folder = '../data/sentence_emd_full'
audio_embeddings_folder = '../data/mfcc'
text_emb_train = load_embeddings(train_df['FileName'].tolist(), text_embeddings_folder)
text_emb_val = load_embeddings(val_df['FileName'].tolist(), text_embeddings_folder)
text_emb_test = load_embeddings(test_df['FileName'].tolist(), text_embeddings_folder)
audio_emb_train = load_embeddings(train_df['FileName'].tolist(), audio_embeddings_folder, is_audio=True)
audio_emb_val = load_embeddings(val_df['FileName'].tolist(), audio_embeddings_folder, is_audio=True)
audio_emb_test = load_embeddings(test_df['FileName'].tolist(), audio_embeddings_folder, is_audio=True)

text_embeddings_aug_folder = '../data/sentence_emd_full_aug'
audio_embeddings_aug_folder = '../data/mfcc_aug'
text_emb_train_aug = load_embeddings(train_df['FileName'].tolist(), text_embeddings_aug_folder)
text_emb_val_aug = load_embeddings(val_df['FileName'].tolist(), text_embeddings_aug_folder)
text_emb_test_aug = load_embeddings(test_df['FileName'].tolist(), text_embeddings_aug_folder)
audio_emb_train_aug = load_embeddings(train_df['FileName'].tolist(), audio_embeddings_aug_folder, is_audio=True)
audio_emb_val_aug = load_embeddings(val_df['FileName'].tolist(), audio_embeddings_aug_folder, is_audio=True)
audio_emb_test_aug = load_embeddings(test_df['FileName'].tolist(), audio_embeddings_aug_folder, is_audio=True)

# Get labels for train, validation, and test sets
label_list_train = train_df['Label_num'].tolist()
label_list_val = val_df['Label_num'].tolist()
label_list_test = test_df['Label_num'].tolist()
gender_train = extract_gender(train_df['FileName'].tolist())
gender_val = extract_gender(val_df['FileName'].tolist())
gender_test = extract_gender(test_df['FileName'].tolist())

# Define the neural network with Transformer and LSTM
class NeuralNetwork(nn.Module):
    def __init__(self, text_emb_dim, img_emb_dim, audio_emb_dim):
        super(NeuralNetwork, self).__init__()
        self.img_lstm = nn.LSTM(512, 768 // 2, batch_first=True, dropout=0.4, bidirectional=True)
        self.audio_lstm = nn.LSTM(20, 768 // 2, batch_first=True, dropout=0.4, bidirectional=True)
        self.fc1 = nn.Linear(768 * 3, 2000)
        self.fc2 = nn.Linear(2000, 1000)
        self.fc3_emotion = nn.Linear(1000, 4)
        self.fc3_gender = nn.Linear(1000, 2)
        self.fc_text = nn.Linear(1000, text_emb_dim)
        self.fc_img = nn.Linear(1000, img_emb_dim)
        # self.fc_audio = nn.Linear(1000, audio_emb_dim)
        self.fc_audio = nn.Linear(1000, 768)
        self.dropout = nn.Dropout(0.4)

    def forward(self, text_emb, img_folder, audio_emb, augmented=False):
        img_folder_suffix = '_aug' if augmented else ''
        img_folder_full = f'../data/frames_emd_each_audio{img_folder_suffix}/' + img_folder

        img_embeddings = []
        for img_file in os.listdir(img_folder_full):
            if img_file.endswith('.npy'):
                img_emb = np.load(os.path.join(img_folder_full, img_file))
                img_embeddings.append(img_emb)
        img_embeddings = torch.tensor(np.array(img_embeddings), dtype=torch.float32, device=device)
        img_lstm_out_all, _ = self.img_lstm(img_embeddings.unsqueeze(0))
        img_lstm_mean = img_lstm_out_all.mean(dim=1)

        audio_emb = torch.stack([emb.to(device) for emb in audio_emb], dim=0)
        audio_lstm_out_all, _ = self.audio_lstm(audio_emb)
        # print(f"audio earlier: {audio_lstm_out_all.shape}")
        audio_lstm_mean = audio_lstm_out_all.mean(dim=0).unsqueeze(0)
        # print(f"text: {text_emb.shape}, img: {img_lstm_mean.shape}, audio: {audio_lstm_mean.shape}")
        combined_embeddings = torch.cat((text_emb, img_lstm_mean, audio_lstm_mean), dim=1)
        x = self.dropout(torch.nn.functional.leaky_relu(self.fc1(combined_embeddings)))
        x = self.dropout(torch.nn.functional.leaky_relu(self.fc2(x)))

        emotion_output = self.fc3_emotion(x)
        gender_output = self.fc3_gender(x)

        text_output = self.fc_text(x)
        img_output = self.fc_img(x)
        audio_output = self.fc_audio(x)

        return emotion_output, gender_output, x, text_output, img_output, img_lstm_mean, audio_output, audio_lstm_mean


# Create an instance of the neural network
text_emb_dim = text_emb_train[0].shape[0]
img_emb_dim = text_emb_dim  # Assuming img_embeddings are loaded similarly
audio_emb_dim = audio_emb_train[0].shape[0]

# Create an instance of the neural network
fc_model = NeuralNetwork(text_emb_dim, img_emb_dim, audio_emb_dim).to(device)

# Define the loss function and optimizer for the neural network
fc_criterion = nn.CrossEntropyLoss()
contrastive_criterion = BatchContrastiveLoss(margin=1e-1).to(device)
reconstruction_criterion = nn.MSELoss()
fc_optimizer = optim.Adam(fc_model.parameters(), lr=0.002)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(fc_optimizer, mode='min', factor=0.9, patience=5, verbose=True)

batch_size = 500

start_epoch = 0
total_epochs = 70
with open('./results/new_1_mfcc_tsne.txt', 'a') as f:
    f.write("Margin: 0.1.\n")
    for epoch in range(start_epoch, total_epochs):
        fc_model.train()
        running_loss = 0.0

        if epoch == 0 or epoch == total_epochs-1:
            with torch.no_grad():
                
                embeddings_list = []
                label_list = []

                for i in range(0, len(text_emb_test), batch_size):

                    text_inputs = text_emb_test[i:i + batch_size].to(device)
                    audio_inputs = audio_emb_test[i:i + batch_size]
                    text_inputs_aug = text_emb_test_aug[i:i + batch_size].to(device)
                    audio_inputs_aug = audio_emb_test_aug[i:i + batch_size]
                    emotion_labels = torch.tensor(label_list_test[i:i + batch_size]).to(device)
                    gender_labels = gender_test[i:i + batch_size].to(device)
                    img_folders = [name for name in test_df['FileName'].iloc[i:i + batch_size].tolist()]

                    fc_optimizer.zero_grad()
                    batch_loss = 0
                    for j, folder in enumerate(img_folders):
                        emotion_output, gender_output, embeddings, text_output, img_output, img_true, audio_output, audio_true = fc_model(text_inputs[j].unsqueeze(0), folder, audio_inputs[j])
                        emotion_output_aug, gender_output_aug, embeddings_aug, text_output_aug, img_output_aug, img_true_aug, audio_output_aug, audio_true_aug = fc_model(text_inputs_aug[j].unsqueeze(0), folder, audio_inputs_aug[j], augmented=True)
                        
                        embeddings_list.append(embeddings)
                        label = emotion_labels[j]
                        label_list.append(label)
                        embeddings_list.append(embeddings_aug)
                        label = emotion_labels[j]
                        label_list.append(label)

                if epoch == 0:
                    all_embeddings_epoch1 = torch.cat(embeddings_list).detach().cpu().numpy()
                    all_labels_epoch1 = torch.tensor(label_list).cpu().numpy()
                    plot_tsne(all_embeddings_epoch1, all_labels_epoch1, epoch+1, 'tsne_epoch_1_test.png')

                if epoch == total_epochs-1:
                    all_embeddings_epoch1 = torch.cat(embeddings_list).detach().cpu().numpy()
                    all_labels_epoch1 = torch.tensor(label_list).cpu().numpy()
                    plot_tsne(all_embeddings_epoch1, all_labels_epoch1, epoch+1, 'tsne_epoch_70_test.png')

        for i in tqdm(range(0, len(text_emb_train), batch_size), desc=f"Epoch {epoch+1}"):

            embeddings_list = []
            label_list = []

            text_inputs = text_emb_train[i:i + batch_size].to(device)
            audio_inputs = audio_emb_train[i:i + batch_size]
            text_inputs_aug = text_emb_train_aug[i:i + batch_size].to(device)
            audio_inputs_aug = audio_emb_train_aug[i:i + batch_size]
            emotion_labels = torch.tensor(label_list_train[i:i + batch_size]).to(device)
            gender_labels = gender_train[i:i + batch_size].to(device)
            img_folders = [name for name in train_df['FileName'].iloc[i:i + batch_size].tolist()]

            fc_optimizer.zero_grad()
            batch_loss = 0
            for j, folder in enumerate(img_folders):
                emotion_output, gender_output, embeddings, text_output, img_output, img_true, audio_output, audio_true = fc_model(text_inputs[j].unsqueeze(0), folder, audio_inputs[j])
                emotion_output_aug, gender_output_aug, embeddings_aug, text_output_aug, img_output_aug, img_true_aug, audio_output_aug, audio_true_aug = fc_model(text_inputs_aug[j].unsqueeze(0), folder, audio_inputs_aug[j], augmented=True)
                
                embeddings_list.append(embeddings)
                label = emotion_labels[j]
                label_list.append(label)
                embeddings_list.append(embeddings_aug)
                label = emotion_labels[j]
                label_list.append(label)

                loss_emotion = fc_criterion(emotion_output, emotion_labels[j:j+1])
                loss_gender = fc_criterion(gender_output, gender_labels[j:j+1])
                loss_emotion_aug = fc_criterion(emotion_output_aug, emotion_labels[j:j+1])
                loss_gender_aug = fc_criterion(gender_output_aug, gender_labels[j:j+1])

                loss_text_recon = reconstruction_criterion(text_output, text_inputs[j].unsqueeze(0))
                loss_img_recon = reconstruction_criterion(img_output, img_true)
                loss_audio_recon = reconstruction_criterion(audio_output, audio_true)

                loss_text_recon_aug = reconstruction_criterion(text_output_aug, text_inputs_aug[j].unsqueeze(0))
                loss_img_recon_aug = reconstruction_criterion(img_output_aug, img_true_aug)
                loss_audio_recon_aug = reconstruction_criterion(audio_output_aug, audio_true_aug)

                loss = (loss_emotion + loss_gender + loss_emotion_aug + loss_gender_aug + 
                        loss_text_recon + loss_img_recon + loss_audio_recon +
                        loss_text_recon_aug + loss_img_recon_aug + loss_audio_recon_aug)
                batch_loss += loss

            # Calculate contrastive loss after collecting all embeddings in the batch
            all_embeddings = torch.cat(embeddings_list)
            all_labels = torch.tensor(label_list, device=device)
            contrastive_loss = contrastive_criterion(all_embeddings, all_labels)
            batch_loss += contrastive_loss

            batch_loss.backward()
            fc_optimizer.step()
            running_loss += batch_loss.item()

        val_loss = 0.0
        val_emotion_accuracy = 0.0
        val_gender_accuracy = 0.0
        val_emotion_accuracy_aug = 0.0
        val_gender_accuracy_aug = 0.0
        total_val_samples = 0
        fc_model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(text_emb_val), batch_size), desc="Validation"):

                embeddings_list = []
                label_list = []

                text_inputs = text_emb_val[i:i + batch_size].to(device)
                audio_inputs = audio_emb_val[i:i + batch_size]
                text_inputs_aug = text_emb_val_aug[i:i + batch_size].to(device)
                audio_inputs_aug = audio_emb_val_aug[i:i + batch_size]
                emotion_labels = torch.tensor(label_list_val[i:i + batch_size]).to(device)
                gender_labels = gender_val[i:i + batch_size].to(device)
                img_folders = [name for name in val_df['FileName'].iloc[i:i + batch_size].tolist()]

                for j, folder in enumerate(img_folders):
                    emotion_output, gender_output, embeddings, text_output, img_output, img_true, audio_output, audio_true = fc_model(text_inputs[j].unsqueeze(0), folder, audio_inputs[j])
                    emotion_output_aug, gender_output_aug, embeddings_aug, text_output_aug, img_output_aug, img_true_aug, audio_output_aug, audio_true_aug = fc_model(text_inputs_aug[j].unsqueeze(0), folder, audio_inputs_aug[j], augmented=True)
                    
                    embeddings_list.append(embeddings)
                    label = emotion_labels[j]
                    label_list.append(label)
                    embeddings_list.append(embeddings_aug)
                    label = emotion_labels[j]
                    label_list.append(label)

                    loss_emotion = fc_criterion(emotion_output, emotion_labels[j:j+1])
                    loss_gender = fc_criterion(gender_output, gender_labels[j:j+1])
                    loss_emotion_aug = fc_criterion(emotion_output_aug, emotion_labels[j:j+1])
                    loss_gender_aug = fc_criterion(gender_output_aug, gender_labels[j:j+1])

                    loss_text_recon = reconstruction_criterion(text_output, text_inputs[j].unsqueeze(0))
                    loss_img_recon = reconstruction_criterion(img_output, img_true)
                    loss_audio_recon = reconstruction_criterion(audio_output, audio_true)

                    loss_text_recon_aug = reconstruction_criterion(text_output_aug, text_inputs_aug[j].unsqueeze(0))
                    loss_img_recon_aug = reconstruction_criterion(img_output_aug, img_true_aug)
                    loss_audio_recon_aug = reconstruction_criterion(audio_output_aug, audio_true_aug)

                    val_loss += (loss_emotion.item() + loss_gender.item() + 
                                 loss_emotion_aug.item() + loss_gender_aug.item() + 
                                 loss_text_recon.item() + loss_img_recon.item() + loss_audio_recon.item() +
                                 loss_text_recon_aug.item() + loss_img_recon_aug.item() + loss_audio_recon_aug.item())

                    _, emotion_preds = torch.max(emotion_output, 1)
                    _, gender_preds = torch.max(gender_output, 1)
                    _, emotion_preds_aug = torch.max(emotion_output_aug, 1)
                    _, gender_preds_aug = torch.max(gender_output_aug, 1)
                    val_emotion_accuracy += (emotion_preds == emotion_labels[j:j+1]).sum().item()
                    val_gender_accuracy += (gender_preds == gender_labels[j:j+1]).sum().item()
                    val_emotion_accuracy_aug += (emotion_preds_aug == emotion_labels[j:j+1]).sum().item()
                    val_gender_accuracy_aug += (gender_preds_aug == gender_labels[j:j+1]).sum().item()
                    total_val_samples += 1

        if epoch == 0:
            all_embeddings_epoch1 = torch.cat(embeddings_list).detach().cpu().numpy()
            all_labels_epoch1 = torch.tensor(label_list).cpu().numpy()
            plot_tsne(all_embeddings_epoch1, all_labels_epoch1, epoch+1, 'tsne_epoch_1_validation.png')

        if epoch == total_epochs-1:
            all_embeddings_epoch1 = torch.cat(embeddings_list).detach().cpu().numpy()
            all_labels_epoch1 = torch.tensor(label_list).cpu().numpy()
            plot_tsne(all_embeddings_epoch1, all_labels_epoch1, epoch+1, 'tsne_epoch_70_validation.png')
            
        all_embeddings = torch.cat(embeddings_list)
        all_labels = torch.tensor(label_list, device=device)
        contrastive_loss = contrastive_criterion(all_embeddings, all_labels)
        # Calculate average validation loss and step the scheduler
        # val_loss /= len(text_emb_val)
        val_loss += contrastive_loss.item()
        scheduler.step(val_loss)

        val_emotion_accuracy /= total_val_samples
        val_gender_accuracy /= total_val_samples
        val_emotion_accuracy_aug /= total_val_samples
        val_gender_accuracy_aug /= total_val_samples
        train_loss = running_loss / len(text_emb_train)
        val_loss /= total_val_samples
        print(f"Epoch: {epoch + 1},  Training Loss: {train_loss:.4f}")
        f.write(f"Epoch: {epoch + 1}, Training Loss: {train_loss:.4f}\n")
        print(f"Epoch: {epoch + 1}, val Loss: {val_loss:.4f}")
        f.write(f"Epoch: {epoch + 1}, Training Loss: {val_loss:.4f}\n")
        print(f"Validation Emotion Accuracy: {val_emotion_accuracy:.4f}, Validation Gender Accuracy: {val_gender_accuracy:.4f}, Validation Emotion Accuracy Augmented: {val_emotion_accuracy_aug:.4f}, Validation Gender Accuracy Augmented: {val_gender_accuracy_aug:.4f}")
        f.write(f"Validation Emotion Accuracy: {val_emotion_accuracy:.4f}, Validation Gender Accuracy: {val_gender_accuracy:.4f}, Validation Emotion Accuracy Augmented: {val_emotion_accuracy_aug:.4f}, Validation Gender Accuracy Augmented: {val_gender_accuracy_aug:.4f}\n")

    # Test set evaluation
    test_loss = 0.0
    test_emotion_accuracy = 0.0
    test_gender_accuracy = 0.0
    total_test_samples = 0
    emotion_preds_list = []
    emotion_labels_list = []
    gender_preds_list = []
    gender_labels_list = []
    with torch.no_grad():
        for i in tqdm(range(0, len(text_emb_test), batch_size), desc="Testing"):
            text_inputs = text_emb_test[i:i + batch_size].to(device)
            audio_inputs = audio_emb_test[i:i + batch_size]
            emotion_labels = torch.tensor(label_list_test[i:i + batch_size]).to(device)
            gender_labels = gender_test[i:i + batch_size].to(device)
            img_folders = [name for name in test_df['FileName'].iloc[i:i + batch_size].tolist()]

            for j, folder in enumerate(img_folders):
                emotion_output, gender_output, _,_,_,_,_,_= fc_model(text_inputs[j].unsqueeze(0), folder, audio_inputs[j])
                loss_emotion = fc_criterion(emotion_output, emotion_labels[j:j+1])
                loss_gender = fc_criterion(gender_output, gender_labels[j:j+1])
                test_loss += loss_emotion.item() + loss_gender.item()

                _, emotion_preds = torch.max(emotion_output, 1)
                _, gender_preds = torch.max(gender_output, 1)
                test_emotion_accuracy += (emotion_preds == emotion_labels[j:j+1]).sum().item()
                test_gender_accuracy += (gender_preds == gender_labels[j:j+1]).sum().item()
                total_test_samples += 1

                # Collect predictions and labels for confusion matrix
                emotion_preds_list.append(emotion_preds.cpu().numpy())
                emotion_labels_list.append(emotion_labels[j:j+1].cpu().numpy())
                gender_preds_list.append(gender_preds.cpu().numpy())
                gender_labels_list.append(gender_labels[j:j+1].cpu().numpy())

        # Convert lists to numpy arrays for confusion matrix computation
        emotion_preds = np.concatenate(emotion_preds_list)
        emotion_labels = np.concatenate(emotion_labels_list)
        gender_preds = np.concatenate(gender_preds_list)
        gender_labels = np.concatenate(gender_labels_list)

        emotion_cm = confusion_matrix(emotion_labels, emotion_preds)
        gender_cm = confusion_matrix(gender_labels, gender_preds)
        test_emotion_accuracy2 = accuracy_score(emotion_labels, emotion_preds)
        test_gender_accuracy2 = accuracy_score(gender_labels, gender_preds)

        test_emotion_accuracy /= total_test_samples
        test_gender_accuracy /= total_test_samples
        print(f"Test Emotion Accuracy: {test_emotion_accuracy:.4f}, Test Gender Accuracy: {test_gender_accuracy:.4f}")
        f.write(f"Test Emotion Accuracy: {test_emotion_accuracy:.4f}, Test Gender Accuracy: {test_gender_accuracy:.4f}\n")

        print("Emotion Confusion Matrix:\n", emotion_cm)
        print("Gender Confusion Matrix:\n", gender_cm)
        # Optionally, write the confusion matrix to the file
        f.write("Emotion Confusion Matrix:\n" + str(emotion_cm) + "\n")
        f.write("Gender Confusion Matrix:\n" + str(gender_cm) + "\n")

        print(f"Test Emotion Accuracy2: {test_emotion_accuracy2:.4f}, Test Gender Accuracy2: {test_gender_accuracy2:.4f}")
        f.write(f"Test Emotion Accuracy2: {test_emotion_accuracy2:.4f}, Test Gender Accuracy2: {test_gender_accuracy2:.4f}\n")

        report_emotion = classification_report(emotion_labels, emotion_preds, target_names=['Neutral', 'Angry', 'Sad', 'Happy'], digits=5)
        print("Emotion Report:\n", report_emotion)
        f.write("Emotion Report:\n" + str(report_emotion) + "\n")
        # label_mapping = {'neu': 0, 'ang': 1, 'sad': 2, 'hap': 3, 'exc': 3}