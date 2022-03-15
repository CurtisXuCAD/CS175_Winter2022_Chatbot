import sys, os, json, torch
cwd = os.path.join(os.getcwd(),'src')
sys.path.insert(0, cwd)
from run_utils import *

if __name__ == '__main__':
    save_dir = os.path.join("src","save")
    corpus_name = "data_intermediate"
    # corpus_name = "cornell_movie-dialogs_corpus"
    checkpoint_iter = 100
    hidden_size = 500
    encoder_n_layers = 5
    decoder_n_layers = 3
    dropout = 0.1
    batch_size = 64
    
    voc = Voc(corpus_name)

    # Configure models
    attn_model = 'dot'
    #attn_model = 'general'
    #attn_model = 'concat'


    loadFilename = os.path.join(save_dir,  corpus_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size), 
                               '{}_checkpoint.tar'.format(checkpoint_iter))



    # Load model if a loadFilename is provided
    if loadFilename:
        # If loading on same machine the model was trained on
        checkpoint = torch.load(loadFilename)
        # If loading a model trained on GPU to CPU
        #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        voc.__dict__ = checkpoint['voc_dict']

    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, hidden_size)
    if loadFilename:
        embedding.load_state_dict(embedding_sd)
    # Initialize encoder & decoder models
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
    if loadFilename:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Set dropout layers to eval mode
    encoder.eval()
    decoder.eval()

    # Initialize search module
    searcher = GreedySearchDecoder(encoder, decoder)

    # Begin chatting (uncomment and run the following line to begin)
    # evaluateInput(encoder, decoder, searcher, voc)

#-----------------------------------------------------------------------------
    
    intents_dir = os.path.join(save_dir, 'intents.json')
    with open(intents_dir, 'r') as f:
        intents = json.load(f)
        

    p1_data_dir = os.path.join(save_dir, 'data.pth')
    data = torch.load(p1_data_dir)

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data["all_words"]
    tags = data["tags"]
    model_state = data["model_state"]


    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()

    bot_name = "\nBot"
    print("Let's chat! type 'quit' to exit")
    while True:
        sentence = input('\nYou: ')
        sentence_bk = sentence
        if sentence == "quit":
            break

        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X)

        output = model(X)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]
        # print("tag: ", tag)

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item()> 0.85:
            for intent in intents["intents"]:
                if tag == intent["tag"]:
                    #random_response = random.choice(intent["responses"])
                    #print(f"{bot_name}: {random_response}")
                    print(f"{bot_name}: {random.choice(intent['responses'])}", end=" ")


                    if tag == "time":
                        import time, datetime
                        t = time.localtime()
                        current_time = time.strftime("%H:%M", t)
                        print(f"{datetime.date.today()} {current_time}", end="")

                        ################################
                        #Can apply more functionalities/API here
                        ################################
                    
                    print()
        else:
            # print(f"{bot_name}: I do not understand...")
            evaluateInput_API(encoder, decoder, searcher, voc, sentence_bk)