###

Code implementing Simple RNN and LSTM constructions that generate Dyck-(k,m)

As described in the paper

> RNNs can generate bounded hierarchical structure with optimal memory. <br>
> John Hewitt, Michael Hahn, Surya Ganguli, Percy Liang, Christopher D. Manning <br>
> Empirical Methods in Natural Language Processing (EMNLP) 2020. <br>

## Getting started

You must have PyTorch installed.

## Usage

python run_demo.py [rnn_type] [encoding_type] [k] [m]

- rnn_type is one of simplernn, lstm
- encoding type is one of log (for O(m\log k) solution) or linear (for O(mk) solution)
- k is the vocabulary size, must be a power of two (for ease of implementation), and > 1
- m is the maximum nesting depth, must be >0

First, the hidden size of the RNN will be printed out.
Then, the parameter matrices themselves will be printed out. 
We suggest running it for small values of k and m at first, as the matrices themselves give some intuition.

Once running, you will be prompted to input symbol; you must do so one at a time.
The symbols you input must either be open brackets
  
  (a (b (c ... 

(k of them) or close brackets

  a) b) c) ...

(k of them).

So an input might look like
  
  (a [ENTER] # observe output
  (b [ENTER] # observe output
  b) [ENTER] # observe output
  (z [ENTER] # observe output (assuming you set k>=26)


You will be shown the hidden state (for the simple RNN) or the hidden state, cell state and all gate values (for the LSTM.)
You will also be shown the probabilities over the next token as determined by the model.
The probability dictionary will also tell you what symbols you can give to the model next (except END).
Note that END never has to be given to the model; we can just observe the probability of seeing it next at any given prefix.

## Examples

For the simple RNN O(mk) generator,

        python run_demo.py simplernn linear 2 3

For the simple RNN O(m\log k) generator,

        python run_demo.py simplernn log 16 4

For the LSTM O(mk) generator,

        python run_demo.py LSTM linear 8 6

For the LSTM RNN O(m\log k) generator,

        python run_demo.py LSTM log 64 5

# Code layout

 - `run_demo.py`: entrypoint to the demo.
 - `hand_rnn.py`: hand implementations of Simple RNN and LSTM generators
 - `lstm_constructions.py`: code for generating the LSTM constructions for any k,m
 - `simplernn_constructions.py`: code for generating the Simple RNN constructions for any k,m

# Details
For all scaling factors specified in the paper, we use 1e4
