import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

maxPow = 4

csvfile = open('YohaMaruEvent.csv', 'r', encoding='utf8')
dat = csv.reader(csvfile, delimiter=',')
t = []
t1 = []
# put the x coordinates in the list X, the y coordinates in the list Y
for i, row in enumerate(dat):
    if i > 0:
        t.append(i-1.)
        t1.append(float(row[2]))
t = np.array(t)/48
t1 = np.array(t1)/1000
print(t[:10])
print(t1[:10])

# t = t[48:-48]
# t1 = t1[48:-48]

n_observations = len(t)
print(n_observations)
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

def model(X, w):
    terms = []
    for i in range(maxPow):
        term = tf.multiply(w[i], tf.pow(X, i))
        terms.append(term)
    return tf.add_n(terms)
# w = tf.Variable(tf.random_normal([maxPow],mean=10.0, stddev=10.0), name="parameters")
w = tf.Variable([25., 50., -15., 1.6], name="parameters")
Y_pred = model(X,w)
cost = tf.reduce_sum(tf.pow(Y_pred - Y, 2)) / (n_observations - 1)

learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

n_epochs = 1000
with tf.Session() as sess:
    # Here we tell tensorflow that we want to initialize all
    # the variables in the graph so we can use them
    init = tf.global_variables_initializer()
    sess.run(init)

    # Fit all training data
    prev_training_cost = 0.0
    for epoch_i in range(n_epochs):
        for (x, y) in zip(t, t1):
            sess.run(optimizer, feed_dict={X: x, Y: y})
        training_cost = sess.run(cost, feed_dict={X: t, Y: t1})
        # print(training_cost)

        if epoch_i % 100 == 0:
            print('epoch:', epoch_i)
            w.eval()
            w=tf.Print(w, [w[0]], "This is w[0]: ")
            w=tf.Print(w, [w[1]], "This is w[1]: ")
            w=tf.Print(w, [w[2]], "This is w[2]: ")
            w=tf.Print(w, [w[3]], "This is w[3]: ")
            print(training_cost)
            plt.plot(t, Y_pred.eval(feed_dict={X: t}, session=sess), 'k', alpha=epoch_i / n_epochs)

        # Allow the training to quit if we've reached a minimum
        if (np.abs(prev_training_cost - training_cost))/(n_observations-1) <0.000001:#0.000001
            w.eval()
            w=tf.Print(w, [w[0]], "This is w[0]: ")
            w=tf.Print(w, [w[1]], "This is w[1]: ")
            w=tf.Print(w, [w[2]], "This is w[2]: ")
            w=tf.Print(w, [w[3]], "This is w[3]: ")
            print(training_cost)
            plt.plot(t, Y_pred.eval(feed_dict={X: t}, session=sess), 'k', color='teal')
            break

        prev_training_cost = training_cost


    plt.plot(t,t1)
    plt.show()
