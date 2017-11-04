function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.
  visible_data = sample_bernoulli(visible_data);
  hidden_probabilities = visible_state_to_hidden_probabilities(rbm_w, visible_data);
  hidden_states = sample_bernoulli(hidden_probabilities);
  rbm_w_grad = configuration_goodness_gradient(visible_data, hidden_states);

  visible_probabilities = hidden_state_to_visible_probabilities(rbm_w, hidden_states) ;
  visible_states = sample_bernoulli(visible_probabilities);

  %hidden_probabilities = visible_state_to_hidden_probabilities(rbm_w, visible_states)
  %hidden_states = sample_bernoulli(hidden_probabilities)
  %rbm_w_grad = rbm_w_grad .- configuration_goodness_gradient(visible_states, hidden_states)
  
  hidden_probabilities = visible_state_to_hidden_probabilities(rbm_w, visible_states);
  rbm_w_grad = rbm_w_grad .- configuration_goodness_gradient(visible_states, hidden_probabilities);
  
  %ret = configuration_goodness_gradient(visible_states, hidden_states)
  ret = rbm_w_grad;
end
