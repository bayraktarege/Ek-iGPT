# EksiGPT
A pre-trained small GPT model on 8K Ekşi Sözlük (Turkish Urban Dictionary) entries.

You can find the weghts here: https://drive.google.com/drive/folders/180R9E80t0SaTeE2cbqQCbIuXMcNMagze?usp=share_link

Large Language Models reached their highest popularity in November 2022 with the launch of ChatGPT. The Generative Pre-Trained Transformer (GPT) architecture laying behind of this model was described in Vashwani et al. paper titled "Attention Is All You Need". In comparison to the Recurrent Neural Network approach, a transformer connects encoder and decoder to achive shorter training time and better quality in translation.

This is a prototype for a decoder only model capable of text-completion. Model source code belongs to Andrej Karpathy (@karpathy). I used the same caharacter based approach and pre-trained the model for 10000 epochs and reached train and validation loss around 1.34.  

The reason for this project is to understand the transformer mechanism and get a grasp of hyperparameters for further versions. Also there are relatively small amount of open-source Turkish language models capbale of producing reasonable output. Improving this model requires fine-tuning with reinforcement learning. This repository provides you the dataset, over 13K Ekşi Sözlük topics and hopes to give your projects a kickstart in building convenient language models capable of producing high-quality Turkish output.


Further reading:
https://medium.com/deeper-learning/glossary-of-deep-learning-word-embedding-f90c3cec34ca
https://www.youtube.com/watch?v=jUBeXwRBVZM - Twitter Bot
https://huggingface.co/dbmdz/bert-base-turkish-cased - BERT - Turkish Corpus
https://towardsdatascience.com/word2vec-explained-49c52b4ccb71 Word2Vec explained
https://towardsdatascience.com/understanding-language-modelling-nlp-part-1-ulmfit-b557a63a672b - ULMFiT
https://huggingface.co/gorkemgoknar/gpt2-turkish-writer
https://pytorch.org/tutorials/beginner/transformer_tutorial.html#run-the-model
https://towardsdatascience.com/how-to-create-a-chatbot-with-python-deep-learning-in-less-than-an-hour-56a063bdfc44 - chatbot keras
https://www.sciencedirect.com/science/article/pii/S2666827020300062  chatbot types
8428-6471-2211-1466
https://developer.nvidia.com/blog/how-to-create-a-custom-language-model/ - create custom model
https://medium.com/@andelkovics98/the-power-of-embeddings-unraveling-the-secrets-of-natural-language-processing-in-llms-like-gpt-4-601c14d1d0ea - embeddings
https://www.veribilimiokulu.com/natural-language-toolkitnltk/ * nutk tutorial stemming etc
https://huyenchip.com/2023/06/07/generative-ai-strategy.html
https://www.youtube.com/watch?v=kCc8FmEb1nY&t=5630s - 
https://github.com/CoomingSoon/TurkishNLPGuide - türkçe nlp
https://lajavaness.medium.com/multiclass-and-multilabel-text-classification-in-one-bert-model-95c54aab59dc
https://www.youtube.com/watch?v=Q9zv369Ggfk&ab_channel=AIJason : Fine Tune LLM Finding datasets
https://towardsdatascience.com/fine-tuning-large-language-models-llms-23473d763b91 fine tune LLM
https://betterprogramming.pub/unleash-your-digital-twin-how-fine-tuning-llm-can-create-your-perfect-doppelganger-b5913e7dda2e — Digital Twin
Q and a without context https://discuss.huggingface.co/t/bert-question-answering-model-without-context/5093
https://www.reddit.com/r/LocalLLaMA/comments/15sgg4m/what_modules_should_i_target_when_training_using/
LLaMA 2 — https://colab.research.google.com/drive/12dVqXZMIVxGI0uutU6HG9RWbWPXL3vts?usp=sharing burada pet config datasında training modules yok
https://towardsdatascience.com/fine-tuning-large-language-models-llms-23473d763b91 fine tune LLM

