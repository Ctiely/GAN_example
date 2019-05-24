# https://github.com/liber145/rlpack/blob/master/rlpack/algos/base.py

import os
from abc import ABC, abstractmethod
import tensorflow as tf


class Base(ABC):
    """Algorithm base class."""

    def __init__(self, save_path=None):
        # Save.
        self.save_path = save_path

        # ------------------------ Reset graph ------------------------
        tf.reset_default_graph()
        tf.Variable(0, name="global_step", trainable=False)
        self.increment_global_step = tf.assign_add(tf.train.get_global_step(), 1)

        # ------------------------ Build network ------------------------
        self._build_network()

        # ------------------------ Build algorithm ------------------------
        self._build_algorithm()

        # ------------------------ Initialize model store and reload. ------------------------
        self._prepare()

    @abstractmethod
    def _build_network(self):
        """Build tensorflow operations for algorithms."""
        pass

    @abstractmethod
    def _build_algorithm(self):
        """Build algorithms using prebuilt networks."""
        pass

    def _prepare(self):
        # ------------------------ Initialize saver. ------------------------
        self.saver = tf.train.Saver(max_to_keep=5)

        # ------------------------ Initialize Session. ------------------------
        conf = tf.ConfigProto(allow_soft_placement=True)
        conf.gpu_options.allow_growth = True  # pylint: disable=E1101
        self.sess = tf.Session(config=conf)

        # ------------------------ Initialize tensorflow variables.  ------------------------
        self.sess.run(tf.global_variables_initializer())

        # ------------------------ Reload model from the saved path. ------------------------
        self.load_model()

    @abstractmethod
    def train(self, datas, training_epoches, batch_size):
        pass

    def save_model(self):
        """Save model to `save_path`."""
        save_dir = os.path.join(self.save_path, "model")
        os.makedirs(save_dir, exist_ok=True)
        global_step = self.sess.run(tf.train.get_global_step())
        self.saver.save(
            self.sess,
            os.path.join(save_dir, "model"),
            global_step,
            write_meta_graph=True
        )

    def load_model(self):
        """Load model from `save_path` if there exists."""
        latest_checkpoint = tf.train.latest_checkpoint(os.path.join(self.save_path, "model"))
        if latest_checkpoint:
            print("## Loading model checkpoint {} ...".format(latest_checkpoint))
            self.saver.restore(self.sess, latest_checkpoint)
        else:
            print("## New start!")