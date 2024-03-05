This is a just a little example of a neural network in c#.  It's not perfect, but it shows the basics of how a fully connected network works.

If you want to create a network of a different size, you'll have to do the following:
Example:
var structure = new int[] { 6, 8, 6, 4, 3 };
var net = new MLP(structure);

This would create a network with 6 features (an array of 6 elements), a hidden layer with 8 neurons, a hidden layer with 6 neurons, a hidden layer with 4 neurons, and an output layer with 3 neurons.  It'll probably even work.

If you want to change the activations, you'll want to change from Sigmoid or Tanh to something else.
