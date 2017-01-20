import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet

# TODO: Load traffic signs data.
training_file = 'train_traffic.p'
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
X_train, y_train = train['features'], train['labels']
nb_classes = max(y_train)+1
print("Data Loaded.", X_train.shape, nb_classes)

# TODO: Split data into training and validation sets.
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.33, random_state=0)

X_train, y_train = shuffle(X_train, y_train)



# TODO: Define placeholders and resize operation.
features = tf.placeholder(tf.float32, (None, 32, 32, 3))
labels = tf.placeholder(tf.int64, (None))
resized = tf.image.resize_images(features, (227, 227))
#one_hot_y = tf.one_hot(y, nb_classes)

print("Final Training   size: ", resized)
print("Final Validation size: ", len(X_validation))

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix
weights8 = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
biases8  = tf.Variable(tf.zeros(nb_classes))
logits   = tf.add(tf.matmul(fc7, weights8), biases8)



# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
BATCH_SIZE = 64
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer()
training_operation = optimizer.minimize(loss_operation, var_list=[weights8, biases8])


# set up training evaluation system
predictions = tf.argmax(logits, 1)
correct_prediction = tf.equal(predictions, labels)
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def evaluate(X_data, y_data, sess):
    num_examples = len(X_data)
    total_accuracy = 0
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        print(batch_x.shape)
        accuracy = sess.run(accuracy_operation, feed_dict={features: batch_x, labels: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples




# TODO: Train and evaluate the feature extraction model.
EPOCHS = 2
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        batch_i = 0
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_i += 1
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            print(batch_x.shape)
            sess.run(training_operation, feed_dict={features: batch_x, labels: batch_y})
            print("Batch: ", batch_i)
            if batch_i > 50:
                break
            
        validation_accuracy = evaluate(X_validation, y_validation, sess)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
