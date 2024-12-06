import tensorflow as tf
from tensorflow.keras import backend as K
from keras import optimizers
from tensorflow.keras.layers import BatchNormalization as BN, Concatenate, Dense, Input, Lambda,Dropout, Conv1DTranspose, Conv1D, Flatten, Reshape
from keras.losses import mean_squared_error,binary_crossentropy
from keras.models import Model
from tensorflow.keras.utils import plot_model


import numpy as np
import networkx as nx
import pandas as pd
from datetime import datetime



import pip
try:
  import torch_geometric
except ModuleNotFoundError:
  pip.main(['install', 'torch_geometric'])
  import torch_geometric

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.loss
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import GCNConv, summary, SAGEConv, GraphConv
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import aggr


class VAE():
  def __init__(self, args):
      self.args = args
      self_vae = None

      if args.vtype == "VAE-CONCAT":
          self.VAE_CONCAT_built()
      elif args.vtype == "VAE-EARLY":
          self.VAE_EARLY_built()
      elif args.vtype == "VAE-INT":
          self.VAE_INT_built()
      elif args.vtype == "VAE-LATE":
          self.VAE_LATE_built()

#VAE-DEFAULT

  def VAE_CONCAT_built(self):

    # Build the encoder network
    # ------------ Input -----------------
    prot_inp = Input(shape=(self.args.inp_prot_size,))

    # ------------ Concat Layer -----------------
    x = Dense(self.args.dense_layer_size, activation="relu")(prot_inp)
    x1 = Dense(self.args.dense_layer_size // 2, activation="relu")(x)

    # ------------Embedding Layer --------------

    z_mean = Dense(self.args.latent_size, name='z_mean')(x1)
    z_log_sigma = Dense(self.args.latent_size, name='z_log_sigma')(x1)
    z = Lambda(self.sampling, output_shape=(self.args.latent_size,), name='z')([z_mean, z_log_sigma])
    self.encoder = Model(prot_inp, z_mean, name='encoder')

    # -------Build the decoder network------------------
    

    x = Dense(self.args.dense_layer_size // 2, activation="relu")(z)
    x = Dense(self.args.dense_layer_size, activation="relu")(x)

    decoder_outputs = Dense(self.args.inp_prot_size, activation="sigmoid")(x)

    # ------------Final Out -----------------------

    self.vae = Model(prot_inp, decoder_outputs, name='VAE-CONCAT')
    distance = tf.reduce_sum(-0.5 * (1 + z_log_sigma - tf.square(z_mean) - tf.exp(z_log_sigma)))
    r_loss = mean_squared_error(prot_inp, decoder_outputs)
    vae_loss = tf.mean(r_loss + (0.0001 * distance))
    self.vae.add_loss(vae_loss)
    # encoder_metric = tf.keras.metrics.MeanSquaredError(name = 'encoder_metric')
    self.vae.compile(optimizer='adam')


#VAE-EARLY --------------------------------------------------------------------------------------------------------------------------------------------------

  def VAE_EARLY_built(self):

    # Build the encoder network
    # ------------ Input -----------------
    prot_inp = Input(shape=(self.args.inp_prot_size,))
    rna_inp = Input(shape=(self.args.inp_rna_size,))
    inputs = [prot_inp, rna_inp]

    # ------------ Concat Layer -----------------

    x1 = Dense(self.args.dense_layer_size, activation="relu")(prot_inp)
    x1 = Dense(self.args.dense_layer_size // 2, activation="relu")(x1)
    
    x2 = Dense(self.args.dense_layer_size, activation="relu")(rna_inp)
    x2 = Dense(self.args.dense_layer_size // 2, activation="relu")(x2)

    x12 = Concatenate(axis=-1)([x1, x2])

    # ------------Embedding Layer --------------

    z_mean = Dense(self.args.latent_size, name='z_mean')(x12)
    z_log_sigma = Dense(self.args.latent_size, name='z_log_sigma')(x12)
    z = Lambda(self.sampling, output_shape=(self.args.latent_size,), name='z')([z_mean, z_log_sigma])
    self.encoder = Model(inputs, z_mean, name='encoder')

    # -------Build the decoder network------------------
    

    x_p = Dense(self.args.dense_layer_size // 2, activation="relu")(z)
    x_p = Dense(self.args.dense_layer_size, activation="relu")(x_p)
    decoder_outputs_prot = Dense(self.args.inp_prot_size, activation="sigmoid")(x_p)
      
    x_r = Dense(self.args.dense_layer_size // 2, activation="relu")(z)
    x_r = Dense(self.args.dense_layer_size, activation="relu")(x_r)
    decoder_outputs_rna = Dense(self.args.inp_rna_size, activation="sigmoid")(x_r)

    # ------------Final Out -----------------------
    outputs = [decoder_outputs_prot, decoder_outputs_rna]
    self.vae = Model(inputs, outputs, name='VAE-EARLY')
    distance = -0.5 * K.sum(1 + z_log_sigma - K.math.square(z_mean) - K.exp(z_log_sigma), axis=-1)
    r_loss_prot = mean_squared_error(prot_inp, decoder_outputs_prot)
    r_loss_rna = mean_squared_error(rna_inp, decoder_outputs_rna)
    vae_loss = K.mean(r_loss_prot + r_loss_rna + (distance))
    # vae_loss = K.mean(r_loss_prot + r_loss_rna)
    self.vae.add_loss(vae_loss)
    # encoder_metric = tf.keras.metrics.MeanSquaredError(name = 'encoder_metric')
    self.vae.compile(optimizer='adam')
    plot_model(self.vae, to_file='early_model.png', show_shapes=True, show_layer_names=True)
    plot_model(self.encoder, to_file='early_encoder.png', show_shapes=True, show_layer_names=True)


#VAE-INTERMEDIATE-----------------------------------------------------------------------------------------------------------------------------

  def VAE_INT_built(self):

    # Build the encoder network
    # ------------ Input -----------------
    prot_inp = Input(shape=(self.args.inp_prot_size,))
    rna_inp = Input(shape=(self.args.inp_rna_size,))
    inputs = [prot_inp, rna_inp]

    # ------------ Concat Layer -----------------

    x1 = Dense(self.args.dense_layer_size, activation="relu")(prot_inp)
    x1 = Dense(self.args.dense_layer_size // 2, activation="relu")(x1)
    
    x2 = Dense(self.args.dense_layer_size, activation="relu")(rna_inp)
    x2 = Dense(self.args.dense_layer_size // 2, activation="relu")(x2)

    # ------------Embedding Layer --------------

    z_mean_prot = Dense(self.args.latent_size, name='z_mean_prot')(x1)
    z_log_sigma_prot = Dense(self.args.latent_size, name='z_log_sigma_prot')(x1)

    z_mean_rna = Dense(self.args.latent_size, name='z_mean_rna')(x2)
    z_log_sigma_rna = Dense(self.args.latent_size, name='z_log_sigma_rna', kernel_initializer='zeros')(x2)

    z_mean_concat = Concatenate(axis=-1)([z_mean_prot, z_mean_rna])
    z_log_sigma_concat = Concatenate(axis=-1)([z_log_sigma_prot, z_log_sigma_rna])

    z = Lambda(self.sampling, output_shape=(self.args.latent_size * 2,), name='z')([z_mean_concat, z_log_sigma_concat])
    self.encoder = Model(inputs, z_mean_concat, name='encoder')

       # -------Build the decoder network------------------
    

    x_p = Dense(self.args.dense_layer_size // 2, activation="relu")(z)
    x_p = Dense(self.args.dense_layer_size, activation="relu")(x_p)
    decoder_outputs_prot = Dense(self.args.inp_prot_size, activation="sigmoid")(x_p)
      
    x_r = Dense(self.args.dense_layer_size // 2, activation="relu")(z)
    x_r = Dense(self.args.dense_layer_size, activation="relu")(x_r)
    decoder_outputs_rna = Dense(self.args.inp_rna_size, activation="sigmoid")(x_r)

    # ------------Final Out -----------------------
    outputs = [decoder_outputs_prot, decoder_outputs_rna]
    self.vae = Model(inputs, outputs, name='VAE-INT')
    distance = -0.5 * K.sum(1 + z_log_sigma_concat - K.square(z_mean_concat) - K.exp(z_log_sigma_concat), axis=-1)
    r_loss_prot = mean_squared_error(prot_inp, decoder_outputs_prot)
    r_loss_rna = mean_squared_error(rna_inp, decoder_outputs_rna)
    vae_loss = K.mean(r_loss_prot + r_loss_rna + (0.0001 * distance))
    self.vae.add_loss(vae_loss)
    # encoder_metric = tf.keras.metrics.MeanSquaredError(name = 'encoder_metric')
    self.vae.compile(optimizer='adam')
    plot_model(self.vae, to_file='int_model.png', show_shapes=True, show_layer_names=True)
    plot_model(self.encoder, to_file='int_encoder.png', show_shapes=True, show_layer_names=True)



#VAE-LATE--------------------------------------------------------------------------------------------------------------------------------

  def VAE_LATE_built(self):

    # Build the encoder network
    # ------------ Input -----------------
    prot_inp = Input(shape=(self.args.inp_prot_size,))
    rna_inp = Input(shape=(self.args.inp_rna_size,))
    inputs = [prot_inp, rna_inp]

    # ------------ Concat Layer -----------------

    x1 = Dense(self.args.dense_layer_size, activation="relu")(prot_inp)
    x1 = Dense(self.args.dense_layer_size // 2, activation="relu")(x1)
    
    x2 = Dense(self.args.dense_layer_size, activation="relu")(rna_inp)
    x2 = Dense(self.args.dense_layer_size // 2, activation="relu")(x2)

    # ------------Embedding Layer --------------

    z_mean_prot = Dense(self.args.latent_size, name='z_mean_prot')(x1)
    z_log_sigma_prot = Dense(self.args.latent_size, name='z_log_sigma_prot')(x1)
    z_prot = Lambda(self.sampling, output_shape=(self.args.latent_size,), name='z_prot')([z_mean_prot, z_log_sigma_prot])

    self.encoder_prot = Model(prot_inp, z_mean_prot, name='encoder_prot')


    z_mean_rna = Dense(self.args.latent_size, name='z_mean_rna')(x2)
    z_log_sigma_rna = Dense(self.args.latent_size, name='z_log_sigma_rna', kernel_initializer='zeros')(x2)
    z_rna = Lambda(self.sampling, output_shape=(self.args.latent_size,), name='z_rna')([z_mean_rna, z_log_sigma_rna])

    self.encoder_rna = Model(rna_inp, z_mean_rna, name='encoder_rna')
    
    # ------------- Common embedding -------------------

    # shared_embedding = Concatenate(axis=-1)([self.encoder_rna(x1)[2], self.encoder_rna(x2)[2]])
    shared_embedding = Concatenate()([z_prot, z_rna])

    shared_embedding = Dense(self.args.latent_size, activation="relu", name = "shared_embedding")(shared_embedding)
    
    self.shared_encoder = Model(inputs, shared_embedding, name='encoder_shared')
    # -------Build the decoder network------------------

    x_p = Dense(self.args.dense_layer_size // 2, activation="relu")(shared_embedding)
    x_p = Dense(self.args.dense_layer_size, activation="relu")(x_p)
    decoder_outputs_prot = Dense(self.args.inp_prot_size, activation="sigmoid")(x_p)
      
    x_r = Dense(self.args.dense_layer_size // 2, activation="relu")(shared_embedding)
    x_r = Dense(self.args.dense_layer_size, activation="relu")(x_r)
    decoder_outputs_rna = Dense(self.args.inp_rna_size, activation="sigmoid")(x_r)

    # ------------Final Out -----------------------
    outputs = [decoder_outputs_prot, decoder_outputs_rna]
    self.vae = Model(inputs, outputs, name='VAE-LATE')
    distance_prot = -0.5 * K.sum(1 + z_log_sigma_prot - K.square(z_mean_prot) - K.exp(z_log_sigma_prot), axis=-1)
    distance_rna = -0.5 * K.sum(1 + z_log_sigma_rna - K.square(z_mean_rna) - K.exp(z_log_sigma_rna), axis=-1)
    r_loss_prot = mean_squared_error(prot_inp, decoder_outputs_prot)
    r_loss_rna = mean_squared_error(rna_inp, decoder_outputs_rna)
    vae_loss = K.mean(r_loss_prot + r_loss_rna + ((distance_prot + distance_rna)))
    # vae_loss = K.mean(r_loss_prot + r_loss_rna )
    self.vae.add_loss(vae_loss)
    self.vae.compile(optimizer='adam')
    plot_model(self.vae, to_file='late_model.png', show_shapes=True, show_layer_names=True)
    plot_model(self.encoder_rna, to_file='late_encoder_rna.png', show_shapes=True, show_layer_names=True)
    plot_model(self.encoder_prot, to_file='late_encoder_prot.png', show_shapes=True, show_layer_names=True)
      
      
  # def train(self, s1_train, s2_train = None):
  #   self.vae.fit([s1_train, s2_train], s1_train, epochs=self.args.epochs, batch_size=self.args.batch_size, verbose = 1, shuffle=False)
          
  # def predict(self, s1_data, s2_data = None):
  #   z_mean_prot, _, z_prot = self.encoder_prot.predict(s1_data)
  #   z_mean_rna, _, z_rna = self.encoder_rna.predict(s2_data)
  #   concatenated_latent = np.concatenate([z_mean_prot, z_mean_rna], axis=1)
    
  #   return concatenated_latent

  def train(self, s1_train, s2_train = None):
      if self.args.vtype == "VAE-CONCAT":
          self.vae.fit([s1_train], s1_train, epochs=self.args.epochs, batch_size=self.args.batch_size, verbose = 1, shuffle=True)
      else:
          self.vae.fit([s1_train, s2_train], [s1_train, s2_train], epochs=self.args.epochs, batch_size=self.args.batch_size, verbose = 1, shuffle=False)

  # def predict(self, s1_data, s2_data):
  #     return self.vae.predict([s1_data, s2_data], batch_size=self.args.batch_size, verbose = 0)

  def predict(self, s1_data, s2_data = None):
      if self.args.vtype == "VAE-EARLY" or self.args.vtype == "VAE-INT":
          return self.encoder.predict([s1_data, s2_data])
      elif self.args.vtype == "VAE-LATE":
        z_mean_prot = self.encoder_prot.predict(s1_data)
        z_mean_rna = self.encoder_rna.predict(s2_data)
        return np.concatenate([z_mean_prot, z_mean_rna], axis=1)
      elif self.args.vtype == "VAE-CONCAT":
          return self.encoder.predict(s1_data)
          # return self.vae.predict(s1_data)

  def kl_regu(self, z_mean,z_log_sigma):
      kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
      kl_loss = K.sum(kl_loss, axis=-1)
      kl_loss *= -0.5
      return kl_loss
  
  def sampling(self, args):
      z_mean, z_log_var = args
      batch = K.shape(z_mean)[0]
      dim = K.int_shape(z_mean)[1]
      epsilon = K.random_normal(shape=(batch, dim))
      return z_mean + K.exp(0.5 * z_log_var) * epsilon

  def custom_weighted_mse(self, inp, outp, importances):
      squared_error = tf.square(inp - outp)
      weighted_squared_error = tf.multiply(importances, squared_error)
      weighted_mean_squared_error = tf.reduce_mean(weighted_squared_error)
      return weighted_mean_squared_error


def kl_loss(mu, logvar, n_nodes):
  logvar = logvar.clamp(max=10)
  return 1 / n_nodes * (-0.5 * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), dim = 1)))

# def recon_loss(preds, loss_type = "pos"):
#   epsilon = 1e-6
#   # clamped_values = torch.clamp(preds, epsilon, 1 - epsilon)
#   clamped_values = preds
#   if loss_type == "pos":
#     return -torch.log(clamped_values + 1e-15).sum()
#   else:
#     print(torch.isnan(clamped_values).any())
#     print((preds> 1).any())
#     print(clamped_values)
#     return -torch.log(1 - clamped_values + 1e-15).sum()

def recon_loss(preds, adj):
  pos_mask = adj < 1
  diff = torch.abs(preds - adj)
  mean_error = diff[pos_mask].mean() + diff[~pos_mask].mean()
  del diff
  torch.cuda.empty_cache()
  return mean_error

  # diff = torch.abs(preds - adj)
  # return diff.mean()
def adversarial_loss(preds, actual):
  
  # loss = torch.nn.MSELoss(reduction = "mean")
  # return loss(preds, actual)
  loss =  F.binary_cross_entropy_with_logits(preds, actual)
  return loss

class GraphVAE(nn.Module):
    def __init__(self, prot_dim, rna_dim, hidden_dim, latent_dim, pretrained = False, p_vgae = None, r_vgae = None):
        super(GraphVAE, self).__init__()
        # self.conv1_p = GCNConv(prot_dim, hidden_dim)#, cached= True)
        # self.conv1_r = GCNConv(rna_dim, hidden_dim)#, cached= True)
        # self.conv2_p = GCNConv(hidden_dim, latent_dim)#, cached= True)
        # self.conv2_r = GCNConv(hidden_dim, latent_dim)#, cached= True)
        # self.conv3_p = GCNConv(hidden_dim, latent_dim)#, cached= True)
        # self.conv3_r = GCNConv(hidden_dim, latent_dim)#, cached= True)
        if pretrained:
          print("pretrained weights are loaded!")
          self.conv1_p = p_vgae.conv1
          self.conv2_p = p_vgae.conv2
          self.conv3_p = p_vgae.conv3
          self.conv1_r = r_vgae.conv1
          self.conv2_r = r_vgae.conv2
          self.conv3_r = r_vgae.conv3
        else:
          self.conv1_p = SAGEConv(prot_dim, hidden_dim)#, cached= True)
          self.conv1_r = SAGEConv(rna_dim, hidden_dim)#, cached= True)
          self.conv2_p = SAGEConv(hidden_dim, latent_dim)#, cached= True)
          self.conv2_r = SAGEConv(hidden_dim, latent_dim)#, cached= True)
          self.conv3_p = SAGEConv(hidden_dim, latent_dim)#, cached= True)
          self.conv3_r = SAGEConv(hidden_dim, latent_dim)#, cached= True)

        # self.conv1_p = nn.Linear(prot_dim, hidden_dim)#, cached= True)
        # self.conv1_r = nn.Linear(rna_dim, hidden_dim)#, cached= True)
        # self.conv2_p = nn.Linear(hidden_dim, latent_dim)#, cached= True)
        # self.conv2_r = nn.Linear(hidden_dim, latent_dim)#, cached= True)
        # self.conv3_p = nn.Linear(hidden_dim, latent_dim)#, cached= True)
        # self.conv3_r = nn.Linear(hidden_dim, latent_dim)#, cached= True)

        # self.conv1_p = GraphConv(prot_dim, hidden_dim)#, cached= True)
        # self.conv1_r = GraphConv(rna_dim, hidden_dim)#, cached= True)
        # self.conv2_p = GraphConv(hidden_dim, latent_dim)#, cached= True)
        # self.conv2_r = GraphConv(hidden_dim, latent_dim)#, cached= True)
        # self.conv3_p = GraphConv(hidden_dim, latent_dim)#, cached= True)
        # self.conv3_r = GraphConv(hidden_dim, latent_dim)#, cached= True)
        
        self.fc_decode = nn.Linear(2 * latent_dim, latent_dim)
        self.fc_decode_last = nn.Linear(latent_dim, int(latent_dim / 2))
        # self.fc_discriminator = nn.Linear(latent_dim, cc_dim)
    
    def set_optimizer(self, params):
      return torch.optim.Adam(params)

    def encode(self, x_p, x_r, edge_index_p, edge_index_r, edge_weights = None):
        x_p = self.conv1_p(x_p, edge_index_p).relu()
        mu_p = self.conv2_p(x_p, edge_index_p)
        logvar_p = self.conv3_p(x_p, edge_index_p)

        x_r = self.conv1_r(x_r, edge_index_r).relu()
        mu_r = self.conv2_r(x_r, edge_index_r)
        logvar_r = self.conv3_r(x_r, edge_index_r)

        # x_p = self.conv1_p(x_p).relu()
        # mu_p = self.conv2_p(x_p)
        # logvar_p = self.conv3_p(x_p)

        # x_r = self.conv1_r(x_r).relu()
        # mu_r = self.conv2_r(x_r)
        # logvar_r = self.conv3_r(x_r)

        # x_p = self.conv1_p(x_p, edge_index_p, edge_weights).relu()
        # mu_p = self.conv2_p(x_p, edge_index_p, edge_weights)
        # logvar_p = self.conv3_p(x_p, edge_index_p, edge_weights)

        # x_r = self.conv1_r(x_r, edge_index_r, edge_weights).relu()
        # mu_r = self.conv2_r(x_r, edge_index_r, edge_weights)
        # logvar_r = self.conv3_r(x_r, edge_index_r, edge_weights)

        return mu_p, mu_r, logvar_p, logvar_r

    def reparameterize(self, mu_p, mu_r, logvar_p, logvar_r):
        std_p = torch.exp(0.5 * logvar_p)
        eps_p = torch.randn_like(std_p)

        std_r = torch.exp(0.5 * logvar_r)
        eps_r = torch.randn_like(std_r)

        return mu_p + eps_p * std_p, mu_r + eps_r * std_r

        # logvar = torch.cat((logvar_p, logvar_r), dim=1)
        # mu = torch.cat((mu_p, mu_r), dim=1)
        # std = torch.exp(0.5 * logvar)
        # eps = torch.randn_like(std)
        # return mu + eps * std, mu
        

    def concat_z(self, z_p, z_r):
      z = torch.cat((z_p, z_r), dim=1)
      z = self.fc_decode(z)
      z = self.fc_decode_last(z)
      # return (torch.sigmoid(z))
      return (z)

    # def decode(self, z, edge_index):
    #   value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
    #   return(torch.sigmoid(value))

    def decode(self, z, edge_index = None):
      # value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
      # return(torch.sigmoid(value))
      eps=1e-8
      norm_z =  z / (torch.norm(z, dim=1, keepdim=True) + eps)
      value = torch.mm(norm_z, norm_z.t())
      # value = value[edge_index[0], edge_index[1]]
      return value
    
    # def discriminate(self, z):
    #   return F.sigmoid(self.fc_discriminator(z))

    def forward(self, x_p, x_r, edge_index_p, edge_index_r, edge_index_u = None, neg_edge_index = None, edge_weights = None):
        mu_p, mu_r, logvar_p, logvar_r = self.encode(x_p, x_r, edge_index_p, edge_index_r, edge_weights)
        z_p, z_r = self.reparameterize(mu_p, mu_r, logvar_p, logvar_r)
        z = self.concat_z(z_p, z_r)
        # print(z)
        # cc_predicts = self.discriminate(z)
        recon_x_p = self.decode(z_p, edge_index_p) #self.decode(z_p, edge_index_p)
        recon_x_r = self.decode(z_r, edge_index_r) #self.decode(z_r, edge_index_r)
        recon_u = self.decode(z)
        # print(recon_u.size(), recon_u)
        # if neg_edge_index == None:
        #   neg_edge_index_p =  negative_sampling(edge_index_p, z_p.size(0), num_neg_samples = 10 * edge_index_p.size(1) )
        #   neg_edge_index_r =  negative_sampling(edge_index_r, z_r.size(0), num_neg_samples = 10 * edge_index_r.size(1) )
        #   neg_edge_index_u =  negative_sampling(edge_index_u, z.size(0), num_neg_samples = 10 * edge_index_u.size(1) )
        # else:
        # neg_edge_index_p =  neg_edge_index
        # neg_edge_index_r =  neg_edge_index 
        # neg_edge_index_u =  neg_edge_index
        # recon_neg_p = None # self.decode(z_p,neg_edge_index_p)
        # recon_neg_r = None # self.decode(z_r,neg_edge_index_r)
        # recon_neg_u = None # self.decode(z,neg_edge_index_u)
        # print(recon_neg_u.size(), recon_neg_u)
        return recon_x_p, recon_x_r, recon_u, mu_p, mu_r, logvar_p, logvar_r, z


class SGVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(SGVAE, self).__init__()

        self.conv1 = SAGEConv(input_dim, hidden_dim)#, cached= True)
        self.conv2 = SAGEConv(hidden_dim, latent_dim)#, cached= True)
        self.conv3 = SAGEConv(hidden_dim, latent_dim)#, cached= True)

        # self.conv1 = nn.Linear(input_dim, hidden_dim)#, cached= True)
        # self.conv2 = nn.Linear(hidden_dim, latent_dim)#, cached= True)
        # self.conv3 = nn.Linear(hidden_dim, latent_dim)#, cached= True)
        
        self.fc_decode = nn.Linear(2 * latent_dim, latent_dim)
        self.fc_decode_last = nn.Linear(latent_dim, int(latent_dim / 2))
    
    def set_optimizer(self, params):
      return torch.optim.Adam(params)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        mu = self.conv2(x, edge_index)
        logvar = self.conv3(x, edge_index)
        # x = self.conv1(x).relu()
        # mu = self.conv2(x)
        # logvar = self.conv3(x)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
      eps=1e-8
      norm_z =  z / (torch.norm(z, dim=1, keepdim=True) + eps)
      value = torch.mm(norm_z, norm_z.t())
      return value

    def forward(self, x, edge_index):
        mu, logvar = self.encode(x, edge_index)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z

class SEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(SEncoder, self).__init__()

        self.fc_decode = nn.Linear(2 * input_dim, input_dim)
        self.fc_decode_last = nn.Linear(input_dim, latent_dim )
    
    def set_optimizer(self, params):
      return torch.optim.Adam(params)

    def encode(self, z_p, z_r):
      z = torch.cat((z_p, z_r), dim=1)
      z = self.fc_decode(z)
      z = self.fc_decode_last(z)
      # return (torch.sigmoid(z))
      return (z)

    def decode(self, z):
      eps=1e-8
      norm_z =  z / (torch.norm(z, dim=1, keepdim=True) + eps)
      value = torch.mm(norm_z, norm_z.t())
      return value

    def forward(self, z_p, z_r):
        z = self.encode(z_p, z_r)
        recon_x = self.decode(z)
        return recon_x, z

class log_writer():
  def __init__(self, path):
    self.writer = SummaryWriter("/content/drive/MyDrive/SCPRO/Experiments/Runs/" + path + "_" + datetime.now().strftime("%d_%m_%Y"))
  def write(self, name, value, epoch):
    self.writer.add_scalar(name, value,epoch)

class VAE_HI(nn.Module):
    def __init__(self, input_dim, latent_dim=100):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder layers
        self.encoder_fc1 = nn.Linear(input_dim * 4, 512)
        self.encoder_fc2 = nn.Linear(512, 256)
        self.encoder_fc3 = nn.Linear(256, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        
        # Decoder layers
        self.decoder_fc1 = nn.Linear(latent_dim, 128)
        self.decoder_fc2 = nn.Linear(128, 256)
        self.decoder_fc3 = nn.Linear(256, 512)
        self.decoder_output = nn.Linear(512, input_dim)
        
    def encode(self, x):
        h1 = F.relu(self.encoder_fc1(x))
        h2 = F.relu(self.encoder_fc2(h1))
        h3 = F.relu(self.encoder_fc3(h2))
        mu = self.fc_mu(h3)
        logvar = self.fc_logvar(h3)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h1 = F.relu(self.decoder_fc1(z))
        h2 = F.relu(self.decoder_fc2(h1))
        h3 = F.relu(self.decoder_fc3(h2))
        output = torch.sigmoid(self.decoder_output(h3))
        return output
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def loss_function(self, recon_x, x1, x2, xs, xn, mu, logvar):
        BCE_1 = F.cosine(recon_x, x1, reduction='sum')
        BCE_2 = F.cosine(recon_x, x2, reduction='sum')
        BCE_S= F.cosine(recon_x, xs, reduction='sum')
        BCE_N = F.cosine(recon_x, xn, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return torch.exp(BCE_1 + BCE_2 + BCE_S - BCE_N) + KLD



class VAE_Encoder(nn.Module):
  def __init__(self, input_dim, hidden_dim, latent_dim):
      super(VAE_Encoder, self).__init__()
      self.fc1 = nn.Linear(input_dim, hidden_dim)
      self.fc2_mean = nn.Linear(hidden_dim, latent_dim)
      self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)
      self.relu = nn.ReLU()

  def forward(self, x):
      h = self.relu(self.fc1(x))
      mean = self.fc2_mean(h)
      logvar = self.fc2_logvar(h)
      return mean, logvar

class VAE_Decoder(nn.Module):
  def __init__(self, latent_dim, hidden_dim, output_dim):
      super(VAE_Decoder, self).__init__()
      self.fc1 = nn.Linear(latent_dim, hidden_dim)
      self.fc2 = nn.Linear(hidden_dim, output_dim)
      self.relu = nn.ReLU()
      self.sigmoid = nn.Sigmoid()

  def forward(self, z):
      h = self.relu(self.fc1(z))
      x_hat = self.sigmoid(self.fc2(h))
      return x_hat
# Define the VAE
class VAE_torch(nn.Module):
  def __init__(self, input_dim, hidden_dim, latent_dim):
      super(VAE_torch, self).__init__()
      self.encoder = VAE_Encoder(input_dim, hidden_dim, latent_dim)
      self.decoder = VAE_Decoder(latent_dim, hidden_dim, input_dim)

  def reparameterize(self, mean, logvar):
      std = torch.exp(0.5 * logvar)
      eps = torch.randn_like(std)
      return mean + eps * std

  def forward(self, x):
      mean, logvar = self.encoder(x)
      z = self.reparameterize(mean, logvar)
      x_hat = self.decoder(z)
      return x_hat, mean, logvar



  def loss_function(self, x, x_hat, mean, logvar):
      recon_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
      kld_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
      return recon_loss + kld_loss





