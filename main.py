# Panda is a library for python.
import pandas as pd
# Scikit-learn is machine learning library, and StratfiedKFold is is K-Fold Cross-Validation class.
from sklearn.model_selection import StratifiedKFold
# accuracy_score, precision_recall_fscore_support,  score, confusion_matrix are evaluation metrics.
from sklearn.metrics import accuracy_score, precision_recall_fscore_support as score, confusion_matrix
# BertTokenizer is used to tokenize text data for input to BERT models.
# BertForSequenceClassification imports transformers library which is used in pre-training.
from transformers import BertTokenizer, BertForSequenceClassification
# Torch is core library for PyTorch. 
import torch
# DataLoader is used to create loaders for loading data efficiently in batches.
# TensorDataset is used to create datasets from tensors, and the tensors are used with DataLoader.
from torch.utils.data import DataLoader, TensorDataset
# Tqdm is the library which can provide a progress bar. 
from tqdm import tqdm
# Matplotlib is the library which can create data cizualization. 
import matplotlib.pyplot as plt
# Seaborn is the library which visualize high-level data plotting.
import seaborn as sns
# Numpy is a fundamental library in Python.
import numpy as np
# Dropout is used in neural networks to prevent overfitting.
from torch.nn import Dropout

# Dataset (CSV file) is read and be stored in data frame which is called dataset.
dataset = pd.read_csv('dataset_final.csv')

# CategoryId is defined and assigned encoded category labels using factorize(). 
# When the factorize are applyed, categoryID might be as follows. Physics -> 0, Mathematics -> 1, Cybersecurity -> 2.
dataset['CategoryId'] = dataset['Category'].factorize()[0]

# target_category is defined and it contains the unique category lavels. 
# Unique lavels will be used for classification. 
target_category = dataset['Category'].unique()

# n_splits is K-Fold Cross-Validation value, it is set to 5. 
# As this value increases, the training data and testing data can be finely divided, which determines the accuracy and performance of this program.
# skf is defined as StrainfieldKFold including shuffle data before splitting (shuffle=True) and setting a random seed (random_state=42).
# it specifies a random seed (42) so that the random shuffling and splitting of the data is the same each time the code is executed.
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# This line initializes the BERT tokenizer using the bert-base-uncased (pre-trained BERT model).
# Setting do_lower_case=True will cause all text to be converted to lowercase during tokenization.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# This code is used to determine and set the PyTorch model and the device, CPU or GPU, on which the data will be processed. 
# It checks if a GPU is available, if not, the CPU is used. 
# If a GPU is available, cuda is specified, otherwise, cpu is specified.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# The BERT model 'bert-base-uncased' is loaded on this line.
# Freeze method can be also used BERT model pre-training, but by calling the pre-trained model here, the pre-trained model can be used without the need to Freeze it.
# BERT model has a hidden size of 768.
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(dataset['Category'].unique()), hidden_size=768)
# If the computer can use GPU, this line indicates moving BERT model to GPU. 
model.to(device)
# The dropout rate is set to 0.2 so that the program randomly drops a fraction of neurons to prevent overfitting.
model.dropout = Dropout(0.2)

# This line changes the learning rate for the optimizer. The value was decided by Manami, and the change will affect to this program performance. 
# lr=5e-5 sets the learning rate to 0.00005. 
# weight_decay=1e-4 sets weight decay which helpsprevent overfitting.
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)

avg_val_accuracy = 0.0

# avg_conf_matrix is initialized as a zeros matrix with dimentions. 
avg_conf_matrix = np.zeros((len(target_category), len(target_category)), dtype=int)

# These lines just set empty lists, and the each variable is as follows. 
# training losses
all_train_losses = []
# validation losses
all_val_losses = []
# training accuracies
all_train_accuracies = []
# validation accuracies
all_val_accuracies = []

# Epochs are set to 10 at the end. (It was most accurate value.)
epochs = 10

# From this line, the loop used in the code and the purpose will be explained. 
'''
 This loops repeats cross-validation folds k time. 
 
 1. The Fold number is printed to show what fold the program's on. 
 2. Train_texts and val_texts represent the text data, and train_labels and val_labels represent the category labels. 
 3. Train_encoding and val_encoding are dictionaries that contain input_ids and attention_mask. 
    input_ids is as a list of integer IDs. Each integer corresponds to a specific token in the BERT vocabulary.
    attention_mask shows which tokens in the sequence are actual data (1) and which are padding tokens (0).
    train_texts_tolist() and val_texts.tolist() convert a pandas series containing text data into a regular Python list.
 4. PyTorch tensors are converted labels by this line. It moves them to GPU and CPU during training and evaluation.
 5. These line create train_dataset and val_dataset and they correspond train_dataloader and val_dataloader. 
    Tokenized input IDs and masks lavels are used to create dataset objects. 
    train_detaloader and val_dataloader include train/validation dataset, batch size, shuffle. 
    DataLoader batches and shuffles the data during training. 
    Validation data is not shuffled to ensure consistent evaluation so that val_encodings's shuffle=False. 
 6. These lines just seet empty lists.
'''

for fold, (train_index, val_index) in enumerate(skf.split(dataset['Text'], dataset['CategoryId'])):
    # 1
    print(f"Fold {fold + 1}/{n_splits}")
    
    # 2
    train_texts, val_texts = dataset['Text'].iloc[train_index], dataset['Text'].iloc[val_index]
    train_labels, val_labels = dataset['CategoryId'].iloc[train_index], dataset['CategoryId'].iloc[val_index]
    
    # 3
    train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True)
    val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True)
    
    # 4 
    labels = torch.tensor(train_labels.values).to(device)
    val_labels = torch.tensor(val_labels.values).to(device)
    
    # 5
    train_dataset = TensorDataset(
        torch.tensor(train_encodings['input_ids']).to(device),
        torch.tensor(train_encodings['attention_mask']).to(device),
        labels
    )
    val_dataset = TensorDataset(
        torch.tensor(val_encodings['input_ids']).to(device),
        torch.tensor(val_encodings['attention_mask']).to(device),
        val_labels
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # 6
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_train_accuracies = []
    epoch_val_accuracies = []
    
    '''
    This loop is for one epoch of training. 
    1. This lines call model_train() and it ensures layers including dropout and batch normalization are active during training. 
    Because some somelayers behave differently during training. 
    2. This line initializes empty variables. 
    '''
    
    for epoch in range(epochs):
        
        # 1
        model.train()
        
        # 2
        total_loss = 0.0
        correct_train = 0
        total_samples_train = 0
        
        '''
        This loop is for each batch of training. 
        Batch can process data in chuncks to reduce memory consumption and speed up training.
        1. input_ids, attention_mask, labels can be got from the current batch. 
        2. This line shows that the gradients of the model's parameters are set to zero.
           Because the gradients have to be reset at the beggining of each batch. 
        3. The model provides output containing various information such as loss, accuracy etc. 
           Loss is how well the model's predictions match. Output extract losses. 
           AdamW updates the model's parameters based on caluculation gradients so that it effects training the model possitively. 
        4. The total_loss will be used to calculate the average training loss.
           Predicstion is defined as torch.argmax(). torch.argmax computes the indicies of the max values along a specified dimension of a tensor. 
        '''

        for batch in tqdm(train_dataloader):
            
            # 1
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            
            # 2
            optimizer.zero_grad()
            
            # 3
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            # 4
            total_loss += loss.item()
            predictions = torch.argmax(outputs.logits, dim=1)
            correct_train += (predictions == labels).sum().item()
            total_samples_train += len(labels)
            
        '''
        1. train_loss is computed by dividing the total loss by the total number of training sample. 
           train_loss is the loss of all batches during training. 
           total_sample_train is the total number of training samples processed during the epoch. 
        2. training_accuracy is computed by dividing the the count of correct predictions.
           correct_train is a count of the number of correct predictions which is got during training 
           tlen(train_dataloader.dataset) is the total number of samples in the training dataset. 
           
        Again, training loss is how well the model is learning and training accuracy is the percentage of making accurate predictions on the training data. 
        
        3. model_eval() change the model to evaluation mode.
        4. These lines initialize variables for total loss and accuracy. 
           Like total_loss, correct_train, total_samples_train were used during training, these initializations will be used for the validation phase.
        '''
        
        # 1
        train_loss = total_loss / total_samples_train
        # 2
        train_accuracy = correct_train / len(train_dataloader.dataset)
        
        # 3
        model.eval()
        
        # 4
        total_loss = 0.0
        correct_val = 0
        total_samples_val = 0
        
        '''
        This loop iterates throught the batches. 
        1. input_ids, attention_mask, labels are extracted from the the current batch. 
           After that, these are moved to GPU and CPU using to(devidce).
        2. torch.no_grad() is used to disable gradient computation.
           Basically, gradient computation is not needed anymore because the model evaluation on the validation data was started.
           Outputs are defined as the input_ids and attention_mask. 
           Loss is extracted from outputs to present the error associated with the model's predictions on the current batch of validation data. 
           total_loss adds the validation loss. 
           The predictions give the predicted class labels for each sample in the batch. 
           To compare the predicted labels with the labels counts the correct predictions in the current batch. 
           total_samples_val adds the the total number of validation samples in the current batch. 
        
        '''

        for batch in tqdm(val_dataloader):
            
            # 1
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            
            # 2
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

                predictions = torch.argmax(outputs.logits, dim=1)
                correct_val += (predictions == labels).sum().item()
                total_samples_val += len(labels)
                
        '''
        1. val_loss is computed by dividing the total validation loss by the total number of validations samples. 
           val_accuracy is computed by dividing the count of correct predictions during validations by the total number of samples in the validation dataset. 
        2. These lines print each current epoch, Training accuracy which computed before, and validation accuracy which computed before as well.
        '''
        # 1
        val_loss = total_loss / total_samples_val
        val_accuracy = correct_val / len(val_dataloader.dataset)
        
        epoch_train_losses.append(train_loss)
        epoch_val_losses.append(val_loss)
        epoch_train_accuracies.append(train_accuracy)
        epoch_val_accuracies.append(val_accuracy)
        
        # 2
        print(f'Epoch {epoch + 1}/{epochs}:')
        print(f'Training Accuracy: {train_accuracy * 100:.2f}%')
        print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')

    all_train_losses.append(epoch_train_losses)
    all_val_losses.append(epoch_val_losses)
    all_train_accuracies.append(epoch_train_accuracies)
    all_val_accuracies.append(epoch_val_accuracies)
    
    '''
    1. The initialized empty list predictions can storethe model's predicted class labels. 
    2. graident calculations were disabled for all operations cuz it was stopped before, and in this time, the gradient caluculations are not needed as well. 
    
    The loop iterates over the batches in the validation dataloader. 
    3. It is really similar to what was done in the training loop, the inputIids, attention_mask and labels are moved to the GPU or CPU.
    4. Taiking the argmax of logits along dimention dim=1 obtains the predicted class lavels. 
    5. This line computes the accuracy, precision, recall and F1-score. 
       Accuracy is the overall correctness of the model's predictions. 
       Precision is the accuracy of the positive predictions. 
       Recall is the model's ability to identify all relevant instancdes of the positive class.
       F1-score si the balance of precision and recall. 
    6. Each avg_val_accuracy, and avg_train_accuracy, is updated computing the mean of the each accuracy. 
       avg_cof_matrix is computed ti the running sum store in avg_vonf_matrix. avg_vonf_matrix is defined at the beggining of this program. 
    '''
    model.eval()
    # 1
    predictions = []
    # 2
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            # 3
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            # 4
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
            
        # 5
        val_accuracy = accuracy_score(val_labels.cpu().numpy(), predictions)
        precision, recall, f1_score, _ = score(val_labels.cpu().numpy(), predictions, average='weighted')
        
    # 6
    avg_val_accuracy = np.mean(np.array(all_val_accuracies))
    avg_train_accuracy = np.mean(np.array(all_train_accuracies))
    avg_conf_matrix += confusion_matrix(val_labels.cpu().numpy(), predictions, labels=np.arange(len(target_category)))

# These lines print average training accuracy, average validation accuracy, Precision, Recall, F1-score. 
print(f'Average Training Accuracy Across Folds: {avg_train_accuracy * 100:.2f}%')
print(f'\nAverage Validation Accuracy Across Folds: {avg_val_accuracy * 100:.2f}%')
print(f'Precision: {precision * 100:.2f}')
print(f'Recall: {recall * 100:.2f}')
print(f'F1-score: {f1_score * 100:.2f}')

# The average confusin matrix was computed accros all folds, so this line visualize the model performance
avg_conf_matrix = avg_conf_matrix.astype('float') / avg_conf_matrix.sum(axis=1)[:, np.newaxis]

# These lines are for plotting average confusion matrix. 
# This image is plotted using orrage color, and title and values are plotted as well.
# x-axis is predicted category and y-axis is actual category. 
plt.figure(figsize=(12, 6))
sns.set(font_scale=1.2)  
sns.heatmap(avg_conf_matrix, annot=True, fmt='.2f', cmap='YlOrRd', xticklabels=target_category, yticklabels=target_category)
plt.title('Average Confusion Matrix')
plt.xlabel('Predicted Category')
plt.ylabel('Actual Category')
plt.show()

# These are for checking accuracy and loss in the experiments. 
# The code computes the average loss and accuracy of each epoch for both training and testing. 
# And then, it plots the both loss and accuracy. 

avg_train_losses = np.mean(np.array(all_train_losses), axis=0)
avg_val_losses = np.mean(np.array(all_val_losses), axis=0)
avg_train_accuracies = np.mean(np.array(all_train_accuracies), axis=0)
avg_val_accuracies = np.mean(np.array(all_val_accuracies), axis=0)

# plt.figure(figsize=(12, 6))
# plt.plot(np.arange(1, epochs + 1), avg_train_accuracies, label='Average Train Accuracy')
# plt.plot(np.arange(1, epochs + 1), avg_val_accuracies, label='Average Validation Accuracy')
# plt.title('Model Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(loc='upper left')
# plt.show()

# plt.figure(figsize=(12, 6))
# plt.plot(np.arange(1, epochs + 1), avg_train_losses, label='Average Train Loss')
# plt.plot(np.arange(1, epochs + 1), avg_val_losses, label='Average Validation Loss')
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(loc='upper right')
# plt.show()

# These lines are for experiments as well. 
# As to make sure, the abstract which the category is not given were prepared, and after the program training and testing, it will test the if the program can predict correctly. 

example_abstracts = [
    # Cybersecurity
    "The topic of scientific works on the implementation of modern technologies and systems of automated management of the enterprise, its resources and technical means is analyzed, and the insufficient completeness of research on the features of the integrated approach to the design and deployment of innovative means of production order support. Based on the determined factors of the operation of the enterprise in the latest conditions of the fourth industrial revolution, directions for the formation of strategies for the introduction of the elements of Industry 4.0 in modern printing enterprises, as well as information protection systems, are determined with electronic document circulation. The mechanisms of decision of tasks of management informative risks considered in complex control system by printeries in the conditions of vagueness and at co-operation of elements of control system between itself. The necessity of using a web portal for the formation of printing orders is substantiate, the main components are define and the levels of access to them described. The paper examines the use of classic and gray fuzzy cognitive maps to solve the problem of cyber security risk assessment of the intelligent management system of a printing enterprise. It is demonstrate that the average estimate of local risk, which is formed using an ensemble of two heterogeneous fuzzy cognitive maps, decreases compared to the use of individual cognitive maps. In order tî better, highlight the results of the research, an example of the application of the proposed methodology for assessing the risks of ensuring the integrity of telemetric information in the industrial network of the intelligent technological process management system of a printing enterprise given, with the continuity of the technological process of manufacturing printing products. In addition to the classic FCM, the paradigms of two variants of the FCM extension were also use in the study, namely, the gray FCM, which used to solve the problem of assessing cyber security risks of intelligent management systems of printing enterprises. An analysis of the possibility of building FCM ensembles to increase the effectiveness of risk assessment using several options for formalizing the expert’s knowledge and experience performed. A fragment of the enterprise management system was consider and an analysis of possible directions of attacks on the printing enterprise by malicious software was perform. These are attacks such as replacing the executable files of server and ARM software, overwriting PLC projects during system operation, and refusing to service the equipment. Based on the formed list of attack vectors and the consequences of their implementation, the task of analyzing the risks of cyber security of a printing enterprise, taking into account the impact on the system of possible internal threats, was considered, using the cognitive modeling apparatus as a modeling tool. The scenario of cognitive modeling of the influence of an internal criminal who exploits the vulnerabilities of the software and hardware components of the control system using the given variants of FCM construction is considered. The average assessment of local risks, which formed using an ensemble of cognitive maps, is better from the point of view of dispersion of assessments of the state of target concepts than the use of individual FCMs. The spread of estimates of the state of ensemble concepts is smaller than the spread of estimates of their gray values using the GFCM, on average by 1.4–1.8 times, which indicates a decrease in the influence of the subjectivity factor on the results of risk assessment. The performed scenario modeling showed that the use of the specified means of protection and organizational measures allows reducing the assessment of local risks by 12–18%, which is a significant indicator. This technique allows obtaining a qualitative and quantitative assessment of risk indicators, taking into account the entire set of objective and subjective factors of uncertainty.",
    # Mathematics
    "We examine several types of visibility graphs: bar and semi bar k-visibility graphs, arc and circle k-visibility graphs, and compact vis- ibility graphs. We improve the upper bound on the thickness of bar k-visibility graphs from 2k(9k − 1) to 6k, and prove that the upper bound must be at least k + 1. We also show that the upper bound on the thickness of semi bar k-visibility graphs is between ⌈ 32 (k + 1)⌉ and 2k. We find bounds on the number of edges and the chromatic number of arc and circle k-visibility graphs. Finally, we relate two conjectures on compact visibility graphs, prove that every n-partite graph in the form K1,a1,a2,...,an−1 is a compact visibility graph, and classify all (but one) graphs with at most six vertices as compact visibility graphs.",
    # Physics
    "Neutron techniques nowadays are fundamental in materials science, biomedical, geology, energy, as well as archaeology. Neutron scattering, neutron diffraction, neutron imaging, neutron spectroscopy etc. are examples of techniques that have been developed by researchers to exploit the unique characteristics of neutrons. Neutrons allow us to probe structures, dynamics, and magnetic properties of materials down to the atomic scale - neutrons, which have the property of not being electrically charged, can pass through a material and reveal information of the material itself, as a consequence of their interactions with the atomic nuclei.  Because of their sensitivity to light elements such as hydrogen, lithium and oxygen they have also attracted research attention for studying improved energy materials for nuclear, solar and wind power."
]
# the tokenizer and encoding are used like before when I mentioned and explained detail. 

example_encodings = tokenizer(example_abstracts, truncation=True, padding=True, return_tensors='pt')
# After that the tokenized input IDs and attention masks are moved to the GPU or CPU. 
example_input_ids = example_encodings['input_ids'].to(device)
example_attention_mask = example_encodings['attention_mask'].to(device)

# These functions and methods are same as the previous code, but these lines generate predictions for the example input IDs with attention masks. 
with torch.no_grad():
    model.eval()
    outputs = model(example_input_ids, attention_mask=example_attention_mask)
    predictions = torch.argmax(outputs.logits, dim=1)

# the predicted category are obtained for each example by selecting the catefory with the highest logit score.
predicted_categories = [target_category[label] for label in predictions.cpu().numpy()]

# Finally, the output is printed.
for i, abstract in enumerate(example_abstracts):
    print(f"\nExample {i + 1} - Predicted Category: {predicted_categories[i]}")
