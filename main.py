from utils import config, log, miscellaneous, seed
import os
import numpy as np
import basic_trainer
from NeuroMax.NeuroMax import NeuroMax
import evaluations
import datasethandler
import scipy
import torch

RESULT_DIR = 'results'
DATA_DIR = 'datasets'

if __name__ == "__main__":
    parser = config.new_parser()
    config.add_dataset_argument(parser)
    config.add_model_argument(parser)
    config.add_training_argument(parser)
    args = parser.parse_args()

    current_time = miscellaneous.get_current_datetime()
    current_run_dir = os.path.join(RESULT_DIR, current_time)
    miscellaneous.create_folder_if_not_exist(current_run_dir)

    config.save_config(args, os.path.join(current_run_dir, 'config.txt'))
    seed.seedEverything(args.seed)
    print(args)

    if args.dataset in ['YahooAnswers']:
        read_labels = True
    else:
        read_labels = False

    # load a preprocessed dataset
    dataset = datasethandler.BasicDatasetHandler(
        os.path.join(DATA_DIR, args.dataset), device=args.device, read_labels=read_labels,
        as_tensor=True, contextual_embed=True)

    # create a model
    pretrainWE = scipy.sparse.load_npz(os.path.join(
        DATA_DIR, args.dataset, "word_embeddings.npz")).toarray()

    model = NeuroMax(vocab_size=dataset.vocab_size,
                    num_topics=args.num_topics,
                    num_groups=args.num_groups,
                    dropout=args.dropout,
                    pretrained_WE=pretrainWE if args.use_pretrainWE else None,
                    weight_loss_GR=args.weight_GR,
                    weight_loss_ECR=args.weight_ECR,
                    alpha_ECR=args.alpha_ECR,
                    alpha_GR=args.alpha_GR,
                    weight_loss_InfoNCE=args.weight_InfoNCE,
                    beta_temp=args.beta_temp)
    model.weight_loss_GR = args.weight_GR
    model.weight_loss_ECR = args.weight_ECR
    model = model.to(args.device)

    # create a trainer
    trainer = basic_trainer.BasicTrainer(model, epochs=args.epochs,
                                            learning_rate=args.lr,
                                            batch_size=args.batch_size,
                                            lr_scheduler=args.lr_scheduler,
                                            lr_step_size=args.lr_step_size)


    # train the model
    trainer.train(dataset)

    # save beta, theta and top words
    beta = trainer.save_beta(current_run_dir)
    train_theta, test_theta = trainer.save_theta(dataset, current_run_dir)
    top_words_10 = trainer.save_top_words(
        dataset.vocab, 10, current_run_dir)
    top_words_15 = trainer.save_top_words(
        dataset.vocab, 15, current_run_dir)
    top_words_20 = trainer.save_top_words(
        dataset.vocab, 20, current_run_dir)
    top_words_25 = trainer.save_top_words(
        dataset.vocab, 25, current_run_dir)

    # argmax of train and test theta
    train_theta_argmax = train_theta.argmax(axis=1)
    test_theta_argmax = test_theta.argmax(axis=1)        

    TD_15 = evaluations.compute_topic_diversity(
        top_words_15, _type="TD")
    print(f"TD_15: {TD_15:.5f}")


    # evaluating clustering
    if read_labels:
        clustering_results = evaluations.evaluate_clustering(
            test_theta, dataset.test_labels)
        print(f"NMI: ", clustering_results['NMI'])
        print(f'Purity: ', clustering_results['Purity'])


    TC_15_list, TC_15 = evaluations.topic_coherence.TC_on_wikipedia(
        os.path.join(current_run_dir, 'top_words_15.txt'))
    print(f"TC_15: {TC_15:.5f}")
