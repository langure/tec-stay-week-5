# Week 5: Neural Networks
A neural network is a type of machine learning model inspired by the human brain, consisting of interconnected layers of nodes, or "neurons". These models are used to recognize complex patterns and make decisions.

Each neuron receives input from the neurons in the previous layer, applies a function (usually non-linear) to the weighted sum of its inputs, and passes the result to the neurons in the next layer. This process continues layer by layer until it reaches the output layer, where the final prediction is made.

The magic of neural networks lies in their ability to learn the optimal weights for each input during training. Training involves presenting the network with input data and desired output, and iteratively adjusting the weights and biases to minimize the discrepancy between the predicted and actual output.

Training a neural network is typically done via a process known as gradient descent, a type of stochastic optimization method. The term 'stochastic' implies that the optimization is influenced by random variables. For instance, in stochastic gradient descent (SGD), a common variant, we compute the gradient and update the weights using a single randomly-chosen training example at a time, rather than using the entire training set.

The gradient in gradient descent refers to the derivative of the network's error with respect to the weights. It points in the direction of steepest increase in the error. So, by adjusting the weights in the opposite direction (the direction of steepest decrease), we can reduce the error.

This process is repeated, often thousands or millions of times, each time nudging the weights of the network towards values that reduce the error. With every iteration, the network becomes a slightly better version of itself. After sufficient training, the network will have learned an approximation to the function that maps the input data to the desired output.

Pros and Cons of Neural Networks
Pros:

Versatility and Flexibility: Neural networks can learn to represent a wide range of complex patterns and functions. They can be used for a variety of tasks, from image classification and natural language processing, to recommendation systems and autonomous driving.

Feature Learning: Unlike traditional machine learning models that require hand-engineered features, neural networks can learn useful features directly from raw data.

Cons:

Require Large Amounts of Data: To effectively learn complex patterns, neural networks often require large amounts of labeled training data.

Black Box: Although neural networks can model complex relationships, they lack transparency in their decision-making process and can be challenging to interpret.

Overfitting: Due to their high capacity, neural networks can overfit to the training data if not properly regularized or if they are too complex for the task at hand.

Example: Image Classification
Consider an image classification task where we need to identify whether an image is a cat or a dog. We can train a neural network by feeding it thousands of images of cats and dogs (the more, the better). During training, the network learns to recognize patterns in pixel data that correspond to each type of animal.

After successful training, if we present a new image to the network, it can predict whether it's a cat or a dog. It does this by processing the image through its layers of neurons, each applying its learned weights and biases, ultimately resulting in the final prediction.

# Readings

[Artificial neural networks](https://d1wqtxts1xzle7.cloudfront.net/31377814/Artificial_Neural_Network-libre.pdf?1392407140=&response-content-disposition=inline%3B+filename%3DIISTE_May_30th_Edition_Peer_reviewed_art.pdf&Expires=1691285456&Signature=eB3umMSnbh1JwnQM9fsBHjZZDFEFYi2gSYVk~E3cFhySy~SqPlRHYQhG2pQE5OqEmJ39csE02kRp7y3TRSclIhHYwEOqmCmNBmBV0rEX8CYkBZ8G8gcRV8YcKnOGDNwx6RM3e2dne3dzpyolrEN8SeF8-LsaFrg5pvbrRntXeWTg5eno3Q6UJeRYv0KbO~aCd40L8rPOKVaIsqQ8szQQt1D6t~zP72MWqETxsK-~i6kmtPQJponKEZwd1zqeWruoBEbRIW2VKzjz~sKjc1~Z-S8qhd8eL7BLfaCdsZZxjx1ZWuPasMPtrEVc-dL4NfqjsWUB3Wrbb8bnVo31kHGWyA__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA)


[Introduction to artificial neural networks](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=04d0b6952a4f0c7203577afc9476c2fcab2cba06)

[Book: introduction to neural networks](https://ds.amu.edu.et/xmlui/bitstream/handle/123456789/4338/an-introduction-to-neural-networks.9781857286731.36028.pdf?sequence=1&isAllowed=y)

