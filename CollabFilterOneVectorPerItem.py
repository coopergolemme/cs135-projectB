'''
CollabFilterOneVectorPerItem.py

Defines class: `CollabFilterOneVectorPerItem`

Scroll down to __main__ to see a usage example.
'''

# Make sure you use the autograd version of numpy (which we named 'ag_np')
# to do all the loss calculations, since automatic gradients are needed
import autograd.numpy as ag_np

# Use helper packages
from AbstractBaseCollabFilterSGD import AbstractBaseCollabFilterSGD
from train_valid_test_loader import load_train_valid_test_datasets

import matplotlib.pyplot as plt

# Some packages you might need (uncomment as necessary)
## import pandas as pd
## import matplotlib

# No other imports specific to ML (e.g. scikit) needed!

class CollabFilterOneVectorPerItem(AbstractBaseCollabFilterSGD):
    ''' One-vector-per-user, one-vector-per-item recommendation model.

    Assumes each user, each item has learned vector of size `n_factors`.

    Attributes required in param_dict
    ---------------------------------
    mu : 1D array of size (1,)
    b_per_user : 1D array, size n_users
    c_per_item : 1D array, size n_items
    U : 2D array, size n_users x n_factors
    V : 2D array, size n_items x n_factors

    Notes
    -----
    Inherits *__init__** constructor from AbstractBaseCollabFilterSGD.
    Inherits *fit* method from AbstractBaseCollabFilterSGD.
    '''

    def init_parameter_dict(self, n_users, n_items, train_tuple):
        ''' Initialize parameter dictionary attribute for this instance.

        Post Condition
        --------------
        Updates the following attributes of this instance:
        * param_dict : dict
            Keys are string names of parameters
            Values are *numpy arrays* of parameter values
        '''
        random_state = self.random_state # inherited RandomState object

        # TODO fix the lines below to have right dimensionality & values
        # TIP: use self.n_factors to access number of hidden dimensions
        print("n_users", n_users)
        print("n_items", n_items)
        self.param_dict = dict(
            mu=ag_np.ones(10), # do not need to fix this
            b_per_user=10 + ag_np.ones(n_users), # FIX dimensionality
            c_per_item=10 + ag_np.ones(n_items), # FIX dimensionality
            U=random_state.randn(n_users, self.n_factors), # FIX dimensionality
            V=random_state.randn(n_items, self.n_factors), # FIX dimensionality
            )


    def predict(self, user_id_N, item_id_N,
                mu=None, b_per_user=None, c_per_item=None, U=None, V=None):
        ''' Predict ratings at specific user_id, item_id pairs

        Args
        ----
        user_id_N : 1D array, size n_examples
            Specific user_id values to use to make predictions
        item_id_N : 1D array, size n_examples
            Specific item_id values to use to make predictions
            Each entry is paired with the corresponding entry of user_id_N

        Returns
        -------
        yhat_N : 1D array, size n_examples
            Scalar predicted ratings, one per provided example.
            Entry n is for the n-th pair of user_id, item_id values provided.
        '''
        N = user_id_N.size


        b_i = b_per_user[user_id_N]
        c_i = c_per_item[item_id_N]

        u_for_id = U[user_id_N]
        v_for_id = V[item_id_N]

        yhat_N = mu + b_i + c_i + ag_np.sum(u_for_id * v_for_id, axis=1)   
        assert yhat_N.shape == (N,)

        return yhat_N


    def calc_loss_wrt_parameter_dict(self, param_dict, data_tuple):
        ''' Compute loss at given parameters

        Args
        ----
        param_dict : dict
            Keys are string names of parameters
            Values are *numpy arrays* of parameter values

        Returns
        -------
        loss : float scalar
        '''

        # This is the rating
        y_N = data_tuple[2]

        # This is the rating we predict
        yhat_N = self.predict(data_tuple[0], data_tuple[1], **param_dict)

        sum_square_err = ag_np.sum(ag_np.square(y_N - yhat_N))


        regularization = self.alpha * (
            ag_np.sum(ag_np.square(param_dict['U'])) +
            ag_np.sum(ag_np.square(param_dict['V']))
        )
        

        loss_total = regularization + sum_square_err

        return loss_total    


if __name__ == '__main__':

    # Load the dataset
    train_tuple, valid_tuple, test_tuple, n_users, n_items = \
        load_train_valid_test_datasets()
    
    # Create the model and initialize its parameters
    # to have right scale as the dataset (right num users and items)
    model_k2 = CollabFilterOneVectorPerItem(
        n_epochs=10, batch_size=10000, step_size=0.1,
        n_factors=2, alpha=0.0)
    model_k2.init_parameter_dict(n_users, n_items, train_tuple)
    model_k2.fit(train_tuple, valid_tuple)

    model_k5 = CollabFilterOneVectorPerItem(
        n_epochs=10, batch_size=10000, step_size=0.1,
        n_factors=5, alpha=0.0)
    model_k5.init_parameter_dict(n_users, n_items, train_tuple)
    model_k5.fit(train_tuple, valid_tuple)

    model_k10 = CollabFilterOneVectorPerItem(
        n_epochs=10, batch_size=10000, step_size=0.1,
        n_factors=10, alpha=0.0)
    model_k10.init_parameter_dict(n_users, n_items, train_tuple)
    model_k10.fit(train_tuple, valid_tuple)




