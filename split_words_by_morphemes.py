import morfessor as m
import argparse
import sys
import io

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains a morfessor model given a corpus, and splits all words in the corpus according to the morfessor model. You can also provide a Morfessor binary model file.')
    parser.add_argument('corpus', metavar='corpus', type=str,
                   help='The text corpus, one sentence per line')
    parser.add_argument('-model', metavar='model', type=str,default = None,
                   help='Morfessor model file path (binary file)')
    parser.add_argument('-output_corpus', metavar='output_corpus', type=str, default=None,
                   help='The output corpus file path')
    parser.add_argument('-save_model', metavar='save_model', type=str, default=None,
                   help='Saves the model as a binary model file (provide file path)')
    parser.add_argument('-maxepochs', metavar='maxepochs', type=int, default=None,
                   help='Maximum iterations for training the model (default= no maximum, stop untill convergence)')
    args = parser.parse_args()
    mio = m.io.MorfessorIO()
    
    corpus = mio.read_corpus_file(args.corpus)
    
    # If a model is provided, load the model
    if args.model: 
        sys.stderr.write("reading Morfessor model...\n")
        model = mio.read_any_model(args.model)
    # If not, train, and possibly save the model
    else:
        sys.stderr.write("-- training Morfessor model --\n")
        sys.stderr.write("reading corpus\n")
        model = m.baseline.BaselineModel()
        model.load_data(corpus)
        sys.stderr.write("training model\n")
        if args.maxepochs:
            sys.stderr.write('max epochs:' + str(args.maxepochs)+"\n")
            model.train_batch(max_epochs=args.maxepochs)
        else:
            model.train_batch()
        if args.save_model:
            sys.stderr.write("writing model to file:"+(args.save_model)+"\n")
            mio.write_binary_model_file(args.save_model, model)

    corpus = [s.split() for s in io.open(args.corpus, 'r').read().split('\n') if s!=""]        
    
    output = ""
    
    for sentence in corpus:
        out = []
        for word in sentence:
            try:
                segmentation = model.segment(word)
            except:
                try:
                    segmentation = model.viterbi_segment(word)[0]
                except:
                    segmentation = [word]
            for segment in segmentation:
                out += segmentation
        if args.output_corpus:
            output += (' '.join(out) + '\n')
        else:
            print ' '.join(out).encode('utf-8')
            
    if args.output_corpus:
        sys.stderr.write('writing segmented corpus to'+args.output_corpus+'\n')
        fout = open(args.output_corpus, 'w')
        fout.write(output.encode('utf-8'))
        fout.close()
        
